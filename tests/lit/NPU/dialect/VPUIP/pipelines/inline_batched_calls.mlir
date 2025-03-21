//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch% allow-custom-values=true" --split-input-file -inline %s | FileCheck %s
// REQUIRES: arch-NPU40XX
module @CallChain {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    //CHECK-NOT: func.func private @cmx_declare_buffer
    func.func private @cmx_declare_buffer(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        // original input
        %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        %2 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%3 : !VPURT.Barrier) {
            %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x3x64x64xf16, @DDR>) outputs(%1 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        }
        VPURT.Task waits(%3 : !VPURT.Barrier) {
            %7 = VPUIP.NNDMA {port = 1 : i64} inputs(%1 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x3x64x64xf16, [@CMX_NN, 1]>) -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        }
        VPURT.Task updates(%3 : !VPURT.Barrier) {
            %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%2 : memref<1x3x64x64xf16, [@CMX_NN, 1]>) outputs(%0 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>
    }

// CHECK-LABEL: @cmx_declare_buffer_main
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x64x64xf16>,
func.func @cmx_declare_buffer_main(%arg0: tensor<2x3x64x64xf16>, %arg1: tensor<2x3x64x64xf16>) -> tensor<2x3x64x64xf16> {
    %farg0 = VPURT.DeclareBuffer <DDR> <131072> -> memref<1x3x64x64xf16, @DDR>
    %farg1 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>

    %filler1 = VPURT.DeclareBuffer <DDR> <98304> -> memref<1x3x64x64xf16, @DDR>
    %filler0 = VPURT.DeclareBuffer <DDR> <420352> -> memref<1x3x64x64xf16, @DDR>

    %farg0_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %farg1_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %call_0_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %call_1_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%farg0_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 1 : i64} inputs(%filler0 : memref<1x3x64x64xf16, @DDR>) outputs(%farg0 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task updates(%farg1_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 0 : i64} inputs(%filler1 : memref<1x3x64x64xf16, @DDR>) outputs(%farg1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    VPURT.Task waits(%farg0_barrier_done, %farg1_barrier_done : !VPURT.Barrier, !VPURT.Barrier) updates(%call_0_barrier_done : !VPURT.Barrier) {
      %57 = func.call @cmx_declare_buffer(%farg0, %farg1) {debatched = [0, 2], reordering = 1} : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task waits(%call_0_barrier_done : !VPURT.Barrier) updates(%farg0_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 0 : i64} inputs(%filler0 : memref<1x3x64x64xf16, @DDR>) outputs(%farg0 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task updates(%farg1_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 1 : i64} inputs(%filler1 : memref<1x3x64x64xf16, @DDR>) outputs(%farg1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    VPURT.Task waits(%farg0_barrier_done, %farg1_barrier_done : !VPURT.Barrier, !VPURT.Barrier) updates(%call_1_barrier_done : !VPURT.Barrier) {
      %57 = func.call @cmx_declare_buffer(%farg0, %farg1) {debatched = [1, 2], reordering = 1} : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    return %arg0 : tensor<2x3x64x64xf16>

        // CHECK:  [[SLICE_0_VAR_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK:  [[SLICE_0_VAR_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        // CHECK:  [[SLICE_0_VAR_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK:  VPURT.Task waits([[UNKN_VAR:%.+]], [[UNKN_VAR:%.+]] : !VPURT.Barrier, !VPURT.Barrier) updates([[SLICE_0_VAR_2]] : !VPURT.Barrier) {
        // CHECK:       VPUIP.NNDMA {port = 0 : i64} inputs([[UNKN_VAR:%.+]] : memref<1x3x64x64xf16, @DDR>) outputs([[SLICE_0_VAR_0]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>) -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK:  }
        // CHECK:  VPURT.Task waits([[SLICE_0_VAR_2]] : !VPURT.Barrier) updates([[UNKN_VAR:%.+]]: !VPURT.Barrier) {
        // CHECK:       VPUIP.NNDMA {port = 1 : i64} inputs([[SLICE_0_VAR_0]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs([[SLICE_0_VAR_1]] : memref<1x3x64x64xf16, [@CMX_NN, 1]>) -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        // CHECK:  }
        // CHECK:  VPURT.Task waits([[UNKN_VAR:%.+]], [[UNKN_VAR:%.+]] : !VPURT.Barrier, !VPURT.Barrier) updates([[SLICE_0_VAR_2]] : !VPURT.Barrier) {
        // CHECK:       VPUIP.NNDMA {port = 0 : i64} inputs([[SLICE_0_VAR_1]] : memref<1x3x64x64xf16, [@CMX_NN, 1]>) outputs([[UNKN_VAR:%.+]] : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  }
        // CHECK:  [[SLICE_1_VAR_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 3]>
        // CHECK:   [[SLICE_1_VAR_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [4] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 4]>
        // CHECK:   [[SLICE_1_VAR_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK:  VPURT.Task waits([[UNKN_VAR:%.+]], [[UNKN_VAR:%.+]] : !VPURT.Barrier, !VPURT.Barrier) updates([[SLICE_1_VAR_2]] : !VPURT.Barrier) {
        // CHECK:       VPUIP.NNDMA {port = 0 : i64} inputs([[UNKN_VAR:%.+]] : memref<1x3x64x64xf16, @DDR>) outputs([[SLICE_1_VAR_0]] : memref<1x3x64x64xf16, [@CMX_NN, 3]>) -> memref<1x3x64x64xf16, [@CMX_NN, 3]>
        // CHECK:   }
        // CHECK:  VPURT.Task waits([[SLICE_1_VAR_2]] : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) {
        // CHECK:       VPUIP.NNDMA {port = 1 : i64} inputs([[SLICE_1_VAR_0]] : memref<1x3x64x64xf16, [@CMX_NN, 3]>) outputs([[SLICE_1_VAR_1]] : memref<1x3x64x64xf16, [@CMX_NN, 4]>) -> memref<1x3x64x64xf16, [@CMX_NN, 4]>
        // CHECK:   }
        // CHECK:  VPURT.Task waits([[UNKN_VAR:%.+]], [[UNKN_VAR:%.+]] : !VPURT.Barrier, !VPURT.Barrier) updates([[SLICE_1_VAR_2]] : !VPURT.Barrier) {
        // CHECK:       VPUIP.NNDMA {port = 0 : i64} inputs([[SLICE_1_VAR_1]] : memref<1x3x64x64xf16, [@CMX_NN, 4]>) outputs([[UNKN_VAR:%.+]] : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        // CHECK:   }
    // CHECK: return [[ARG0]] : tensor<2x3x64x64xf16>
}
}

// -----

module @DoNotReorderCMX {
    // Unlike the previous case, no CMX reordering is done here.
    // The transformation by reordering CMX numbers is performing on a function call
    // which has both "debatched = [<partitioning>]" and "reordering = 1" attributes.
    // As no "reordering" attribute is injected, no any reordering is expected
    IE.TileResource 2 of @NCE at 1.700000e+03 MHz

    //CHECK-NOT: func.func private @cmx_declare_buffer
    func.func private @cmx_declare_buffer(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        // original input
        %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<1x3x64x64xf16, @DDR>
        %1 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        %2 = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%3 : !VPURT.Barrier) {
            %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x3x64x64xf16, @DDR>) outputs(%1 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        }
        VPURT.Task waits(%3 : !VPURT.Barrier) {
            %7 = VPUIP.NNDMA {port = 1 : i64} inputs(%1 : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs(%2 : memref<1x3x64x64xf16, [@CMX_NN, 1]>) -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        }
        VPURT.Task updates(%3 : !VPURT.Barrier) {
            %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%2 : memref<1x3x64x64xf16, [@CMX_NN, 1]>) outputs(%0 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>
    }

// CHECK-LABEL: @cmx_declare_buffer_main
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x64x64xf16>,
func.func @cmx_declare_buffer_main(%arg0: tensor<2x3x64x64xf16>, %arg1: tensor<2x3x64x64xf16>) -> tensor<2x3x64x64xf16> {
    %farg0 = VPURT.DeclareBuffer <DDR> <131072> -> memref<1x3x64x64xf16, @DDR>
    %farg1 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>

    %filler1 = VPURT.DeclareBuffer <DDR> <98304> -> memref<1x3x64x64xf16, @DDR>
    %filler0 = VPURT.DeclareBuffer <DDR> <420352> -> memref<1x3x64x64xf16, @DDR>

    %farg0_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %farg1_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %call_0_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %call_1_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%farg0_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 1 : i64} inputs(%filler0 : memref<1x3x64x64xf16, @DDR>) outputs(%farg0 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task updates(%farg1_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 0 : i64} inputs(%filler1 : memref<1x3x64x64xf16, @DDR>) outputs(%farg1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    VPURT.Task waits(%farg0_barrier_done, %farg1_barrier_done : !VPURT.Barrier, !VPURT.Barrier) updates(%call_0_barrier_done : !VPURT.Barrier) {
      %57 = func.call @cmx_declare_buffer(%farg0, %farg1) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task waits(%call_0_barrier_done : !VPURT.Barrier) updates(%farg0_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 0 : i64} inputs(%filler0 : memref<1x3x64x64xf16, @DDR>) outputs(%farg0 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task updates(%farg1_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 1 : i64} inputs(%filler1 : memref<1x3x64x64xf16, @DDR>) outputs(%farg1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    VPURT.Task waits(%farg0_barrier_done, %farg1_barrier_done : !VPURT.Barrier, !VPURT.Barrier) updates(%call_1_barrier_done : !VPURT.Barrier) {
      %57 = func.call @cmx_declare_buffer(%farg0, %farg1) : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    return %arg0 : tensor<2x3x64x64xf16>

        // CHECK:  [[SLICE_0_VAR_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK:  [[SLICE_0_VAR_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        // CHECK:  [[SLICE_0_VAR_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK:  VPURT.Task waits([[UNKN_VAR:%.+]], [[UNKN_VAR:%.+]] : !VPURT.Barrier, !VPURT.Barrier) updates([[SLICE_0_VAR_2]] : !VPURT.Barrier) {
        // CHECK:       VPUIP.NNDMA {port = 0 : i64} inputs([[UNKN_VAR:%.+]] : memref<1x3x64x64xf16, @DDR>) outputs([[SLICE_0_VAR_0]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>) -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK:  }
        // CHECK:  VPURT.Task waits([[SLICE_0_VAR_2]] : !VPURT.Barrier) updates([[UNKN_VAR:%.+]]: !VPURT.Barrier) {
        // CHECK:       VPUIP.NNDMA {port = 1 : i64} inputs([[SLICE_0_VAR_0]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs([[SLICE_0_VAR_1]] : memref<1x3x64x64xf16, [@CMX_NN, 1]>) -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        // CHECK:  }
        // CHECK:  VPURT.Task waits([[UNKN_VAR:%.+]], [[UNKN_VAR:%.+]] : !VPURT.Barrier, !VPURT.Barrier) updates([[SLICE_0_VAR_2]] : !VPURT.Barrier) {
        // CHECK:       VPUIP.NNDMA {port = 0 : i64} inputs([[SLICE_0_VAR_1]] : memref<1x3x64x64xf16, [@CMX_NN, 1]>) outputs([[UNKN_VAR:%.+]] : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  }
        // CHECK:  [[SLICE_1_VAR_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK:   [[SLICE_1_VAR_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <0> -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        // CHECK:   [[SLICE_1_VAR_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK:  VPURT.Task waits([[UNKN_VAR:%.+]], [[UNKN_VAR:%.+]] : !VPURT.Barrier, !VPURT.Barrier) updates([[SLICE_1_VAR_2]] : !VPURT.Barrier) {
        // CHECK:       VPUIP.NNDMA {port = 0 : i64} inputs([[UNKN_VAR:%.+]] : memref<1x3x64x64xf16, @DDR>) outputs([[SLICE_1_VAR_0]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>) -> memref<1x3x64x64xf16, [@CMX_NN, 0]>
        // CHECK:   }
        // CHECK:  VPURT.Task waits([[SLICE_1_VAR_2]] : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) {
        // CHECK:       VPUIP.NNDMA {port = 1 : i64} inputs([[SLICE_1_VAR_0]] : memref<1x3x64x64xf16, [@CMX_NN, 0]>) outputs([[SLICE_1_VAR_1]] : memref<1x3x64x64xf16, [@CMX_NN, 1]>) -> memref<1x3x64x64xf16, [@CMX_NN, 1]>
        // CHECK:   }
        // CHECK:  VPURT.Task waits([[UNKN_VAR:%.+]], [[UNKN_VAR:%.+]] : !VPURT.Barrier, !VPURT.Barrier) updates([[SLICE_1_VAR_2]] : !VPURT.Barrier) {
        // CHECK:       VPUIP.NNDMA {port = 0 : i64} inputs([[SLICE_1_VAR_1]] : memref<1x3x64x64xf16, [@CMX_NN, 1]>) outputs([[UNKN_VAR:%.+]] : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        // CHECK:   }
    // CHECK: return [[ARG0]] : tensor<2x3x64x64xf16>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @HaloCMXPartitioningForReorderingTest {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    //CHECK-NOT: func.func private @cmx_iti_buffer
    func.func private @cmx_iti_buffer(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        %8 = VPURT.DeclareBuffer <CMX_NN> [0] <30720> -> !VPUIP.ITIBuffer<
            1x64x4x17xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}, [@CMX_NN, 0],
            inwardHaloRegions = [
                #VPUIP.HaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 16], cluster_id = 0 : i64>
            ],
            outwardHaloRegions = [
                #VPUIP.OutwardHaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 15], cluster_id = 0 : i64, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
                ]>
        ]>
        %9 = VPURT.DeclareBuffer <CMX_NN> [1] <30720> -> !VPUIP.ITIBuffer<
            1x64x4x18xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}, [@CMX_NN, 1],
            inwardHaloRegions = [
                #VPUIP.HaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 0], cluster_id = 1 : i64>,
                #VPUIP.HaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 17], cluster_id = 1 : i64>
            ],
            outwardHaloRegions = [
                #VPUIP.OutwardHaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 1], cluster_id = 1 : i64, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 16], cluster_id = 0 : i64>
                ]>,
                #VPUIP.OutwardHaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 16], cluster_id = 1 : i64, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 0], cluster_id = 2 : i64>
                ]>
        ]>
        %145552 = VPURT.DeclareBuffer <CMX_NN> [0] <39936> -> memref<1x64x4x16xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
        %145548 = VPURT.DeclareBuffer <CMX_NN> [0] <39936> -> memref<1x64x4x16xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>
        %145546 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        %145547 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task waits(%145546 : !VPURT.Barrier) updates(%145547 : !VPURT.Barrier) {
          %254514 = VPUIP.NCEClusterTask {is_superdense, task_type = #VPUIP.nce_task_type<ELTWISE>} input(%145552 : memref<1x64x4x16xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) weights(%145548 :
     memref<1x64x4x16xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) parent_input(%145552 : memref<1x64x4x16xf16, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>, [@CMX_NN, 0]>) parent_output(%8 : !VPUIP.ITIBuffer<
              1x64x4x17xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}, [@CMX_NN, 0],
              inwardHaloRegions = [
                  #VPUIP.HaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 16], cluster_id = 0 : i64>
              ],
              outwardHaloRegions = [
                  #VPUIP.OutwardHaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 15], cluster_id = 0 : i64, inwardHaloRegions = [
                      #VPUIP.HaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
                  ]>
          ]>) output_ITI_buff(%9 : !VPUIP.ITIBuffer<
              1x64x4x18xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}, [@CMX_NN, 1],
              inwardHaloRegions = [
                  #VPUIP.HaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 0], cluster_id = 1 : i64>,
                  #VPUIP.HaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 17], cluster_id = 1 : i64>
              ],
              outwardHaloRegions = [
                  #VPUIP.OutwardHaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 1], cluster_id = 1 : i64, inwardHaloRegions = [
                      #VPUIP.HaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 16], cluster_id = 0 : i64>
                  ]>,
                  #VPUIP.OutwardHaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 16], cluster_id = 1 : i64, inwardHaloRegions = [
                      #VPUIP.HaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 0], cluster_id = 2 : i64>
                  ]>
          ]>) outputs(%8 : !VPUIP.ITIBuffer<
              1x64x4x17xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}, [@CMX_NN, 0],
              inwardHaloRegions = [
                  #VPUIP.HaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 16], cluster_id = 0 : i64>
              ],
              outwardHaloRegions = [
                  #VPUIP.OutwardHaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 15], cluster_id = 0 : i64, inwardHaloRegions = [
                      #VPUIP.HaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
                  ]>
          ]>) -> !VPUIP.ITIBuffer<
              1x64x4x17xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>}, [@CMX_NN, 0],
              inwardHaloRegions = [
                  #VPUIP.HaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 16], cluster_id = 0 : i64>
              ],
              outwardHaloRegions = [
                  #VPUIP.OutwardHaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 15], cluster_id = 0 : i64, inwardHaloRegions = [
                      #VPUIP.HaloRegionAttr<shape = [1, 64, 4, 1], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
                  ]>
          ]> variants : {
            DPUTask {cluster_id = 0 : i64, haloRegions = [#VPUIP.DPUHaloRegionAttr<xStart = 0 : i64, xEnd = 63 : i64, yStart = 15 : i64, yEnd = 15 : i64, zStart = 0 : i64, zEnd = 3 : i64, targetOffset = -7680 : i64, targetClusters = [1], targetWidth = 64 : i64>], inEnd = [15, 3, 63], inStart = [0, 0, 0], mpe_mode = #VPU.mpe_mode<CUBOID_16x16>, outEnd = [15, 3, 63], outStart = [0, 0, 0], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>}
          } PPE : {
            PPETask {ppe = #VPU.PPEStub<>}
          }
        }


        return %arg1 : memref<1x3x64x64xf16, @DDR>
    }

// CHECK-LABEL: @cmx_iti_buffer_main
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x64x64xf16>,
func.func @cmx_iti_buffer_main(%arg0: tensor<2x3x64x64xf16>, %arg1: tensor<2x3x64x64xf16>) -> tensor<2x3x64x64xf16> {
    %farg0 = VPURT.DeclareBuffer <DDR> <131072> -> memref<1x3x64x64xf16, @DDR>
    %farg1 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>

    %filler1 = VPURT.DeclareBuffer <DDR> <98304> -> memref<1x3x64x64xf16, @DDR>
    %filler0 = VPURT.DeclareBuffer <DDR> <420352> -> memref<1x3x64x64xf16, @DDR>

    %farg0_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %farg1_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %call_0_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %call_1_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%farg0_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 1 : i64} inputs(%filler0 : memref<1x3x64x64xf16, @DDR>) outputs(%farg0 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task updates(%farg1_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 0 : i64} inputs(%filler1 : memref<1x3x64x64xf16, @DDR>) outputs(%farg1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    VPURT.Task waits(%farg0_barrier_done, %farg1_barrier_done : !VPURT.Barrier, !VPURT.Barrier) updates(%call_0_barrier_done : !VPURT.Barrier) {
      %57 = func.call @cmx_iti_buffer(%farg0, %farg1) {debatched = [0, 2], reordering = 1} : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task waits(%call_0_barrier_done : !VPURT.Barrier) updates(%farg0_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 0 : i64} inputs(%filler0 : memref<1x3x64x64xf16, @DDR>) outputs(%farg0 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task updates(%farg1_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 1 : i64} inputs(%filler1 : memref<1x3x64x64xf16, @DDR>) outputs(%farg1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    VPURT.Task waits(%farg0_barrier_done, %farg1_barrier_done : !VPURT.Barrier, !VPURT.Barrier) updates(%call_1_barrier_done : !VPURT.Barrier) {
      %57 = func.call @cmx_iti_buffer(%farg0, %farg1) {debatched = [1, 2], reordering = 1} : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    return %arg0 : tensor<2x3x64x64xf16>


        // CHECK:   [[SLICE_0_VAR_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <30720> -> !VPUIP.ITIBuffer<
        // CHECK:   1x64x4x17xf16, {order = #NWCH}, [@CMX_NN, 0],
        // CHECK:       inwardHaloRegions = [
        // CHECK:           #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 0 : i64>
        // CHECK:       ],
        // CHECK:       outwardHaloRegions = [
        // CHECK:           #VPUIP.OutwardHaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 0[[NOT_IMPORTANT_MATCH:.*inwardHaloRegions]] = [
        // CHECK:               #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 1 : i64>
        // CHECK:           ]>
        // CHECK:   ]>
        // CHECK:   [[SLICE_0_VAR_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <30720> -> !VPUIP.ITIBuffer<
        // CHECK:     1x64x4x18xf16, {order = #NWCH}, [@CMX_NN, 1],
        // CHECK:     inwardHaloRegions = [
        // CHECK:         #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 1[[NOT_IMPORTANT_MATCH:.*]]: i64>,
        // CHECK:         #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 1[[NOT_IMPORTANT_MATCH:.*]]: i64>
        // CHECK:     ],
        // CHECK:     outwardHaloRegions = [
        // CHECK:         #VPUIP.OutwardHaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 1[[NOT_IMPORTANT_MATCH:.*inwardHaloRegions]] = [
        // CHECK:             #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 0 : i64>
        // CHECK:         ]>,
        // CHECK:         #VPUIP.OutwardHaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 1[[NOT_IMPORTANT_MATCH:.*inwardHaloRegions]] = [
        // CHECK:             #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 2 : i64>
        // CHECK:         ]>
        // CHECK: ]>
        // CHECK:      [[SLICE_0_VAR_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <39936> -> memref<1x64x4x16xf16, #NHWC, [@CMX_NN, 0]>
        // CHECK:      [[SLICE_0_VAR_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <39936> -> memref<1x64x4x16xf16, #NHWC, [@CMX_NN, 0]>
        // CHECK:  VPURT.Task waits([[UNKN_BARRIER:%.+]] : !VPURT.Barrier) updates([[UNKN_BARRIER:%.+]] : !VPURT.Barrier) {
        // CHECK:         [[UNKN_VAR:%.+]] = VPUIP.NCEClusterTask {[[NOT_IMPORTANT_MATCH:.*]]} input([[SLICE_0_VAR_2]] : memref<1x64x4x16xf16, #NHWC, [@CMX_NN, 0]>) weights([[SLICE_0_VAR_3]] : memref<1x64x4x16xf16, #NHWC, [@CMX_NN, 0]>) parent_input([[SLICE_0_VAR_2]] : memref<1x64x4x16xf16, #NHWC, [@CMX_NN, 0]>) parent_output([[SLICE_0_VAR_0]] : !VPUIP.ITIBuffer<
        // CHECK:      1x64x4x17xf16, {order = #NWCH}, [@CMX_NN, 0],
        // CHECK:      inwardHaloRegions = [
        // CHECK:          #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 0 : i64>
        // CHECK:      ],
        // CHECK:      outwardHaloRegions = [
        // CHECK:          #VPUIP.OutwardHaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 0[[NOT_IMPORTANT_MATCH:.*inwardHaloRegions]] = [
        // CHECK:              #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 1 : i64>
        // CHECK:          ]>
        // CHECK:  ]>) output_ITI_buff([[SLICE_0_VAR_1]] : !VPUIP.ITIBuffer<
        // CHECK:      1x64x4x18xf16, {order = #NWCH}, [@CMX_NN, 1],
        // CHECK:      inwardHaloRegions = [
        // CHECK:          #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 1 : i64>,
        // CHECK:          #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 1 : i64>
        // CHECK:      ],
        // CHECK:      outwardHaloRegions = [
        // CHECK:          #VPUIP.OutwardHaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 1[[NOT_IMPORTANT_MATCH:.*inwardHaloRegions]] = [
        // CHECK:              #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 0 : i64>
        // CHECK:          ]>,
        // CHECK:          #VPUIP.OutwardHaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 1[[NOT_IMPORTANT_MATCH:.*inwardHaloRegions]] = [
        // CHECK:              #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 2 : i64>
        // CHECK:          ]>
        // CHECK:  ]>) outputs([[SLICE_0_VAR_0]] : !VPUIP.ITIBuffer<
        // CHECK:      1x64x4x17xf16, {order = #NWCH}, [@CMX_NN, 0],
        // CHECK:      inwardHaloRegions = [
        // CHECK:          #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 0 : i64>
        // CHECK:      ],
        // CHECK:      outwardHaloRegions = [
        // CHECK:          #VPUIP.OutwardHaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 0[[NOT_IMPORTANT_MATCH:.*inwardHaloRegions]] = [
        // CHECK:              #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 1 : i64>
        // CHECK:          ]>
        // CHECK:  ]>) -> !VPUIP.ITIBuffer<
        // CHECK:      1x64x4x17xf16, {order = #NWCH}, [@CMX_NN, 0],
        // CHECK:      inwardHaloRegions = [
        // CHECK:          #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 0 : i64>
        // CHECK:      ],
        // CHECK:      outwardHaloRegions = [
        // CHECK:          #VPUIP.OutwardHaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 0[[NOT_IMPORTANT_MATCH:.*inwardHaloRegions]] = [
        // CHECK:              #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 1 : i64>
        // CHECK:          ]>
        // CHECK:  ]> variants : {
        // CHECK:    DPUTask {cluster_id = 0 : i64, [[NOT_IMPORTANT_MATCH:.*]]}
        // CHECK:  } PPE : {
        // CHECK:    PPETask {ppe = #VPU.PPEStub<>}
        // CHECK:  }
        // CHECK:}
        // CHECK:  [[SLICE_1_VAR_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <30720> -> !VPUIP.ITIBuffer<
        // CHECK:   1x64x4x17xf16, {order = #NWCH}, [@CMX_NN, 3],
        // CHECK:   inwardHaloRegions = [
        // CHECK:       #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 3 : i64>
        // CHECK:   ],
        // CHECK:   outwardHaloRegions = [
        // CHECK:       #VPUIP.OutwardHaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 3[[NOT_IMPORTANT_MATCH:.*inwardHaloRegions]] = [
        // CHECK:           #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 4 : i64>
        // CHECK:       ]>
        // CHECK:   ]>
        // CHECK:  [[SLICE_1_VAR_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [4] <30720> -> !VPUIP.ITIBuffer<
        // CHECK:    1x64x4x18xf16, {order = #NWCH}, [@CMX_NN, 4],
        // CHECK:    inwardHaloRegions = [
        // CHECK:        #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 4 : i64>,
        // CHECK:        #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 4 : i64>
        // CHECK:    ],
        // CHECK:    outwardHaloRegions = [
        // CHECK:        #VPUIP.OutwardHaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 4[[NOT_IMPORTANT_MATCH:.*inwardHaloRegions]] = [
        // CHECK:            #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 3 : i64>
        // CHECK:        ]>,
        // CHECK:        #VPUIP.OutwardHaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 4[[NOT_IMPORTANT_MATCH:.*inwardHaloRegions]] = [
        // CHECK:            #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 5 : i64>
        // CHECK:        ]>
        // CHECK:   ]>
        // CHECK:  [[SLICE_1_VAR_2:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <39936> -> memref<1x64x4x16xf16, #NHWC, [@CMX_NN, 3]>
        // CHECK:  [[SLICE_1_VAR_3:%.+]] = VPURT.DeclareBuffer <CMX_NN> [3] <39936> -> memref<1x64x4x16xf16, #NHWC, [@CMX_NN, 3]>
        // CHECK:  VPURT.Task waits([[UNKN_VAR:%.+]]: !VPURT.Barrier) updates([[UNKN_VAR:%.+]] : !VPURT.Barrier) {
        // CHECK:  [[UNKN_VAR:%.+]] = VPUIP.NCEClusterTask {[[NOT_IMPORTANT_MATCH:.*]]} input([[SLICE_1_VAR_2]] : memref<1x64x4x16xf16, #NHWC, [@CMX_NN, 3]>) weights([[SLICE_1_VAR_3]] : memref<1x64x4x16xf16, #NHWC, [@CMX_NN, 3]>) parent_input([[SLICE_1_VAR_2]] : memref<1x64x4x16xf16, #NHWC, [@CMX_NN, 3]>) parent_output([[SLICE_1_VAR_0]] : !VPUIP.ITIBuffer<
        // CHECK:      1x64x4x17xf16, {order = #NWCH}, [@CMX_NN, 3],
        // CHECK:      inwardHaloRegions = [
        // CHECK:          #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 3 : i64>
        // CHECK:      ],
        // CHECK:      outwardHaloRegions = [
        // CHECK:          #VPUIP.OutwardHaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 3[[NOT_IMPORTANT_MATCH:.*inwardHaloRegions]] = [
        // CHECK:              #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 4 : i64>
        // CHECK:          ]>
        // CHECK:  ]>) output_ITI_buff([[SLICE_1_VAR_1]] : !VPUIP.ITIBuffer<
        // CHECK:      1x64x4x18xf16, {order = #NWCH}, [@CMX_NN, 4],
        // CHECK:      inwardHaloRegions = [
        // CHECK:          #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 4 : i64>,
        // CHECK:          #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 4 : i64>
        // CHECK:      ],
        // CHECK:      outwardHaloRegions = [
        // CHECK:          #VPUIP.OutwardHaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 4[[NOT_IMPORTANT_MATCH:.*inwardHaloRegions]] = [
        // CHECK:              #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 3 : i64>
        // CHECK:          ]>,
        // CHECK:          #VPUIP.OutwardHaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 4[[NOT_IMPORTANT_MATCH:.*inwardHaloRegions]] = [
        // CHECK:              #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 5 : i64>
        // CHECK:          ]>
        // CHECK:  ]>) outputs([[SLICE_1_VAR_0]] : !VPUIP.ITIBuffer<
        // CHECK:      1x64x4x17xf16, {order = #NWCH}, [@CMX_NN, 3],
        // CHECK:      inwardHaloRegions = [
        // CHECK:          #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 3 : i64>
        // CHECK:      ],
        // CHECK:      outwardHaloRegions = [
        // CHECK:          #VPUIP.OutwardHaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 3[[NOT_IMPORTANT_MATCH:.*inwardHaloRegions]] = [
        // CHECK:              #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 4 : i64>
        // CHECK:          ]>
        // CHECK:  ]>) -> !VPUIP.ITIBuffer<
        // CHECK:      1x64x4x17xf16, {order = #NWCH}, [@CMX_NN, 3],
        // CHECK:      inwardHaloRegions = [
        // CHECK:          #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 3 : i64>
        // CHECK:      ],
        // CHECK:      outwardHaloRegions = [
        // CHECK:          #VPUIP.OutwardHaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 3[[NOT_IMPORTANT_MATCH:.*inwardHaloRegions]] = [
        // CHECK:              #VPUIP.HaloRegionAttr<[[NOT_IMPORTANT_MATCH:.*]], cluster_id = 4 : i64>
        // CHECK:          ]>
        // CHECK:  ]> variants : {
        // CHECK:    DPUTask {cluster_id = 3 : i64, [[NOT_IMPORTANT_MATCH:.*]] : i64>}
        // CHECK:  } PPE : {
        // CHECK:    PPETask {ppe = #VPU.PPEStub<>}
        // CHECK:  }
        // CHECK:}
    // CHECK: return [[ARG0]] : tensor<2x3x64x64xf16>
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
module @DDROffsetFromModuleForReorderingTest {
    IE.TileResource 6 of @NCE at 1.700000e+03 MHz

    //CHECK-NOT: func.func private @ddr_decl_buffer
    func.func private @ddr_decl_buffer(%arg0: memref<1x3x64x64xf16, @DDR>, %arg1: memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR> {
        %0 = VPURT.DeclareBuffer <NetworkInput> [0] <10000> -> memref<1x3x64x64xf16, @DDR>
        %1 = VPURT.DeclareBuffer <DDR> <24347136> -> memref<1x3x64x64xf16, @DDR>
        %2 = VPURT.DeclareBuffer <DDR> <25002496> -> memref<1x3x64x64xf16, @DDR>
        %3 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        VPURT.Task updates(%3 : !VPURT.Barrier) {
            %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x3x64x64xf16, @DDR>) outputs(%1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        VPURT.Task waits(%3 : !VPURT.Barrier) {
            %7 = VPUIP.NNDMA {port = 1 : i64} inputs(%1 : memref<1x3x64x64xf16, @DDR>) outputs(%2 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        VPURT.Task updates(%3 : !VPURT.Barrier) {
            %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%2 : memref<1x3x64x64xf16, @DDR>) outputs(%0 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        }
        return %arg1 : memref<1x3x64x64xf16, @DDR>
    }

// CHECK-LABEL: @ddr_decl_buffer_main
// CHECK-SAME: ([[ARG0:%.+]]: tensor<2x3x64x64xf16>,
func.func @ddr_decl_buffer_main(%arg0: tensor<2x3x64x64xf16>, %arg1: tensor<2x3x64x64xf16>) -> tensor<2x3x64x64xf16> {
    %farg0 = VPURT.DeclareBuffer <DDR> <131072> -> memref<1x3x64x64xf16, @DDR>
    %farg1 = VPURT.DeclareBuffer <DDR> <0> -> memref<1x3x64x64xf16, @DDR>

    %filler1 = VPURT.DeclareBuffer <DDR> <98304> -> memref<1x3x64x64xf16, @DDR>
    %filler0 = VPURT.DeclareBuffer <DDR> <420352> -> memref<1x3x64x64xf16, @DDR>

    %farg0_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %farg1_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %call_0_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %call_1_barrier_done = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%farg0_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 1 : i64} inputs(%filler0 : memref<1x3x64x64xf16, @DDR>) outputs(%farg0 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task updates(%farg1_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 0 : i64} inputs(%filler1 : memref<1x3x64x64xf16, @DDR>) outputs(%farg1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    VPURT.Task waits(%farg0_barrier_done, %farg1_barrier_done : !VPURT.Barrier, !VPURT.Barrier) updates(%call_0_barrier_done : !VPURT.Barrier) {
      %57 = func.call @ddr_decl_buffer(%farg0, %farg1) {debatched = [0, 2], reordering = 1} : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task waits(%call_0_barrier_done : !VPURT.Barrier) updates(%farg0_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 0 : i64} inputs(%filler0 : memref<1x3x64x64xf16, @DDR>) outputs(%farg0 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }

    VPURT.Task updates(%farg1_barrier_done : !VPURT.Barrier) {
      %57 = VPUIP.NNDMA {port = 1 : i64} inputs(%filler1 : memref<1x3x64x64xf16, @DDR>) outputs(%farg1 : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    VPURT.Task waits(%farg0_barrier_done, %farg1_barrier_done : !VPURT.Barrier, !VPURT.Barrier) updates(%call_1_barrier_done : !VPURT.Barrier) {
      %57 = func.call @ddr_decl_buffer(%farg0, %farg1) {debatched = [1, 2], reordering = 1} : (memref<1x3x64x64xf16, @DDR>, memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
    }
    return %arg0 : tensor<2x3x64x64xf16>

        // CHECK:  [[SLICE_0_VAR_0:%.+]] = VPURT.DeclareBuffer <DDR> <24347136> -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  [[SLICE_0_VAR_1:%.+]] = VPURT.DeclareBuffer <DDR> <25002496> -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  [[SLICE_0_VAR_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK:  VPURT.Task waits([[UNKN_VAR:%.+]], [[UNKN_VAR:%.+]] : !VPURT.Barrier, !VPURT.Barrier) updates([[SLICE_0_VAR_2]] : !VPURT.Barrier) {
        // CHECK:       VPUIP.NNDMA {port = 0 : i64} inputs([[UNKN_VAR:%.+]] : memref<1x3x64x64xf16, @DDR>) outputs([[SLICE_0_VAR_0]] : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  }
        // CHECK:  VPURT.Task waits([[SLICE_0_VAR_2]] : !VPURT.Barrier) updates([[UNKN_VAR:%.+]]: !VPURT.Barrier) {
        // CHECK:       VPUIP.NNDMA {port = 1 : i64} inputs([[SLICE_0_VAR_0]] : memref<1x3x64x64xf16, @DDR>) outputs([[SLICE_0_VAR_1]] : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  }
        // CHECK:  [[SLICE_1_VAR_0:%.+]] = VPURT.DeclareBuffer <DDR> <49384208> -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  [[SLICE_1_VAR_1:%.+]] = VPURT.DeclareBuffer <DDR> <50039568> -> memref<1x3x64x64xf16, @DDR>
        // CHECK:  [[SLICE_1_VAR_2:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
        // CHECK:  VPURT.Task waits([[UNKN_VAR:%.+]], [[UNKN_VAR:%.+]] : !VPURT.Barrier, !VPURT.Barrier) updates([[SLICE_1_VAR_2]] : !VPURT.Barrier) {
        // CHECK:       VPUIP.NNDMA {port = 0 : i64} inputs([[UNKN_VAR:%.+]] : memref<1x3x64x64xf16, @DDR>) outputs([[SLICE_1_VAR_0]] : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        // CHECK:   }
        // CHECK:  VPURT.Task waits([[SLICE_1_VAR_2]] : !VPURT.Barrier) updates(%7 : !VPURT.Barrier) {
        // CHECK:       VPUIP.NNDMA {port = 1 : i64} inputs([[SLICE_1_VAR_0]] : memref<1x3x64x64xf16, @DDR>) outputs([[SLICE_1_VAR_1]] : memref<1x3x64x64xf16, @DDR>) -> memref<1x3x64x64xf16, @DDR>
        // CHECK:   }
    // CHECK: return [[ARG0]] : tensor<2x3x64x64xf16>
}
}
