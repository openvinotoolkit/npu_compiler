//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --batch-matmul-to-matmul %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @BatchMatMulToMatMul
module @BatchMatMulToMatMul {

config.Resources 4 of @NCE at 1.850000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

func.func @main() -> () {
    %IN = VPURT.DeclareBuffer <CMX_NN> [0] <2112> -> memref<2x1x16x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    // CHECK:   [[IN_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2112> -> memref<1x16x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[IN_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2240> -> memref<1x16x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %WEIGHTS = VPURT.DeclareBuffer <CMX_NN> [0] <12864> -> memref<2x32x16x1x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHTS_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <12864> -> memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHTS_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <13888> -> memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %OUT = VPURT.DeclareBuffer <CMX_NN> [0] <1600> -> memref<2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1600> -> memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1856> -> memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %WEIGHT_TABLE = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<2x32x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHT_TABLE_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHT_TABLE_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1088> -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    %WAIT_BARRIER = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[WAIT_BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task updates(%WAIT_BARRIER : !VPURT.Barrier) {
        %MATMUL = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            kernel_padding = #VPU.Padding<
                left = 0 : i64,
                right = 0 : i64,
                top = 0 : i64,
                bottom = 0 : i64
            >,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }>
        input(%IN : memref<2x1x16x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        weights(%WEIGHTS : memref<2x32x16x1x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        weight_table(%WEIGHT_TABLE : memref<2x32x1x1x4xsi32, [@CMX_NN, 0]>)
        parent_input(%IN : memref<2x1x16x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        parent_output(%OUT : memref<2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        outputs(%OUT : memref<2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
            -> memref<2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
        variants : {
            DPUTask {
                cluster_id = 0 : i64,
                inEnd = [0, 3, 15],
                inStart = [0, 0, 0],
                mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                outEnd = [0, 3, 31],
                outStart = [0, 0, 0],
                pad = #VPU.Padding<
                    left = 0 : i64,
                    right = 0 : i64,
                    top = 0 : i64,
                    bottom = 0 : i64
                >
            }
        } PPE : {
            PPETask {
                ppe = #VPU.PPEStub<>
            }
        }
    }

    // CHECK:       VPURT.Task updates([[WAIT_BARRIER]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[IN_SLICE_0]] : memref<1x16x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      weights([[WEIGHTS_SLICE_0]] : memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      weight_table([[WEIGHT_TABLE_SLICE_0]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_input([[IN_SLICE_0]] : memref<1x16x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_output([[OUT_SLICE_0]] : memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[OUT_SLICE_0]] : memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK:       VPURT.Task updates([[WAIT_BARRIER]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[IN_SLICE_1]] : memref<1x16x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      weights([[WEIGHTS_SLICE_1]] : memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      weight_table([[WEIGHT_TABLE_SLICE_1]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_input([[IN_SLICE_1]] : memref<1x16x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_output([[OUT_SLICE_1]] : memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[OUT_SLICE_1]] : memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    return
}

}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

!qElemType = !quant.uniform<u8:f16:2, {
    1.0:128, 0.5:128, 0.25:128, 0.125:128,
    1.0:128, 0.5:128, 0.25:128, 0.125:128,
    1.0:128, 0.5:128, 0.25:128, 0.125:128,
    1.0:128, 0.5:128, 0.25:128, 0.125:128
}>

// CHECK-LABEL-DAG: @QuantBatchMatMulToMatMul
// CHECK-DAG:   [[Q_ELEM_TYPE:!.+]] = !quant.uniform<u8:f16:1, {
module @QuantBatchMatMulToMatMul {

config.Resources 4 of @NCE at 1.850000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

func.func @main() -> () {
    %IN = VPURT.DeclareBuffer <CMX_NN> [0] <2112> -> memref<2x1x16x4x1x!qElemType, {order = #GNHWC}, [@CMX_NN, 0]>
    // CHECK:   [[IN_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2112> -> memref<1x16x4x1x[[Q_ELEM_TYPE]], {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[IN_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2176> -> memref<1x16x4x1x[[Q_ELEM_TYPE]], {order = #NHWC}, [@CMX_NN, 0]>
    %WEIGHT = VPURT.DeclareBuffer <CMX_NN> [0] <12864> -> memref<2x32x16x1x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHT_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <12864> -> memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHT_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <13888> -> memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %OUT = VPURT.DeclareBuffer <CMX_NN> [0] <1600> -> memref<2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1600> -> memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1856> -> memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %WEIGHT_TABLE = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<2x32x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHT_TABLE_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHT_TABLE_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1088> -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    %WAIT_BARRIER = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[WAIT_BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task updates(%WAIT_BARRIER : !VPURT.Barrier) {
        %MATMUL = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            kernel_padding = #VPU.Padding<
                left = 0 : i64,
                right = 0 : i64,
                top = 0 : i64,
                bottom = 0 : i64
            >,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }>
        input(%IN : memref<2x1x16x4x1x!qElemType, {order = #GNHWC}, [@CMX_NN, 0]>)
        weights(%WEIGHT : memref<2x32x16x1x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        weight_table(%WEIGHT_TABLE : memref<2x32x1x1x4xsi32, [@CMX_NN, 0]>)
        parent_input(%IN : memref<2x1x16x4x1x!qElemType, {order = #GNHWC}, [@CMX_NN, 0]>)
        parent_output(%OUT : memref<2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        outputs(%OUT : memref<2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
            -> memref<2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
        variants : {
            DPUTask {
                cluster_id = 0 : i64,
                inEnd = [0, 3, 15],
                inStart = [0, 0, 0],
                mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                outEnd = [0, 3, 31],
                outStart = [0, 0, 0],
                pad = #VPU.Padding<
                    left = 0 : i64,
                    right = 0 : i64,
                    top = 0 : i64,
                    bottom = 0 : i64
                >
            }
        } PPE : {
            PPETask {
                ppe = #VPU.PPEStub<>
            }
        }
    }

    // CHECK:       VPURT.Task updates([[WAIT_BARRIER]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[IN_SLICE_0]] : memref<1x16x4x1x[[Q_ELEM_TYPE]], {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      weights([[WEIGHTS_SLICE_0]] : memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      weight_table([[WEIGHT_TABLE_SLICE_0]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_input([[IN_SLICE_0]] : memref<1x16x4x1x[[Q_ELEM_TYPE]], {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_output([[OUT_SLICE_0]] : memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[OUT_SLICE_0]] : memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK:       VPURT.Task updates([[WAIT_BARRIER]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[IN_SLICE_1]] : memref<1x16x4x1x[[Q_ELEM_TYPE]], {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      weights([[WEIGHTS_SLICE_1]] : memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      weight_table([[WEIGHT_TABLE_SLICE_1]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_input([[IN_SLICE_1]] : memref<1x16x4x1x[[Q_ELEM_TYPE]], {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_output([[OUT_SLICE_1]] : memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[OUT_SLICE_1]] : memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    return
}

}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @MatMulWithBatch1ToMatMul
module @MatMulWithBatch1ToMatMul {

config.Resources 4 of @NCE at 1.850000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

func.func @main() -> () {
    %IN = VPURT.DeclareBuffer <CMX_NN> [0] <2112> -> memref<1x1x16x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    // CHECK:   [[IN:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2112> -> memref<1x16x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %WEIGHTS = VPURT.DeclareBuffer <CMX_NN> [0] <12864> -> memref<1x32x16x1x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHTS:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <12864> -> memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %OUT = VPURT.DeclareBuffer <CMX_NN> [0] <1600> -> memref<1x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    // CHECK:   [[OUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1600> -> memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %WEIGHT_TABLE = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<1x32x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHT_TABLE:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    %WAIT_BARRIER = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[WAIT_BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task updates(%WAIT_BARRIER : !VPURT.Barrier) {
        %MATMUL = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            kernel_padding = #VPU.Padding<
                left = 0 : i64,
                right = 0 : i64,
                top = 0 : i64,
                bottom = 0 : i64
            >,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }>
        input(%IN : memref<1x1x16x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        weights(%WEIGHTS : memref<1x32x16x1x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        weight_table(%WEIGHT_TABLE : memref<1x32x1x1x4xsi32, [@CMX_NN, 0]>)
        parent_input(%IN : memref<1x1x16x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        parent_output(%OUT : memref<1x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        outputs(%OUT : memref<1x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
            -> memref<1x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
        variants : {
            DPUTask {
                cluster_id = 0 : i64,
                inEnd = [0, 3, 15],
                inStart = [0, 0, 0],
                mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                outEnd = [0, 3, 31],
                outStart = [0, 0, 0],
                pad = #VPU.Padding<
                    left = 0 : i64,
                    right = 0 : i64,
                    top = 0 : i64,
                    bottom = 0 : i64
                >
            }
        } PPE : {
            PPETask {
                ppe = #VPU.PPEStub<>
            }
        }
    }

    // CHECK:       VPURT.Task updates([[WAIT_BARRIER]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[IN]] : memref<1x16x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      weights([[WEIGHTS]] : memref<32x16x1x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      weight_table([[WEIGHT_TABLE]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_input([[IN]] : memref<1x16x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_output([[OUT]] : memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[OUT]] : memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    return
}

}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @BatchMatMulToMatMulWithItiBuffers
module @BatchMatMulToMatMulWithItiBuffers {

config.Resources 4 of @NCE at 1.850000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

func.func @main() -> () {
    %IN = VPURT.DeclareBuffer <CMX_NN> [0] <2112> -> memref<2x1x16x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    // CHECK:   [[IN_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2112> -> memref<1x16x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[IN_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2240> -> memref<1x16x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %WEIGHTS = VPURT.DeclareBuffer <CMX_NN> [0] <12864> -> memref<2x32x16x1x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    %WEIGHT_TABLE = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<2x32x1x1x4xsi32, [@CMX_NN, 0]>

    %OUT_TILE_0 = VPURT.DeclareBuffer <CMX_NN> [0] <1600> -> !VPUIP.ITIBuffer<
        2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0],
        inwardHaloRegions = [
            #VPUIP.HaloRegionAttr<shape = [2, 1, 16, 4, 1], offset = [0, 0, 16, 0, 0], cluster_id = 0 : i64>
        ],
        outwardHaloRegions = [
            #VPUIP.OutwardHaloRegionAttr<shape = [2, 1, 16, 4, 1], offset = [0, 0, 0, 0, 0], cluster_id = 0 : i64, inwardHaloRegions = [
                #VPUIP.HaloRegionAttr<shape = [2, 1, 16, 4, 1], offset = [0, 0, 0, 0, 0], cluster_id = 1 : i64>
            ]>
        ]>

    %OUT_TILE_1 = VPURT.DeclareBuffer <CMX_NN> [1] <1600> -> !VPUIP.ITIBuffer<
        2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 1],
        inwardHaloRegions = [
            #VPUIP.HaloRegionAttr<shape = [2, 1, 16, 4, 1], offset = [0, 0, 0, 0, 0], cluster_id = 1 : i64>
        ],
        outwardHaloRegions = [
            #VPUIP.OutwardHaloRegionAttr<shape = [2, 1, 16, 4, 1], offset = [0, 0, 16, 0, 0], cluster_id = 1 : i64, inwardHaloRegions = [
                #VPUIP.HaloRegionAttr<shape = [2, 1, 16, 4, 1], offset = [0, 0, 16, 0, 0], cluster_id = 0 : i64>
            ]>
        ]>

    // CHECK:   [[OUT_TILE_0_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1600> -> !VPUIP.ITIBuffer<
    // CHECK:       1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0],
    // CHECK:       inwardHaloRegions = [
    // CHECK:           #VPUIP.HaloRegionAttr<shape = [1, 16, 4, 1], offset = [0, 16, 0, 0], cluster_id = 0 : i64>
    // CHECK:       ],
    // CHECK:       outwardHaloRegions = [
    // CHECK:           #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 4, 1], offset = [0, 0, 0, 0], cluster_id = 0 : i64, inwardHaloRegions = [
    // CHECK:               #VPUIP.HaloRegionAttr<shape = [1, 16, 4, 1], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
    // CHECK:           ]>
    // CHECK:   ]>

    // CHECK:   [[OUT_TILE_0_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1856> -> !VPUIP.ITIBuffer<
    // CHECK:       1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0],

    // CHECK:   [[OUT_TILE_1_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <1600> -> !VPUIP.ITIBuffer<
    // CHECK:       1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 1],
    // CHECK:       inwardHaloRegions = [
    // CHECK:           #VPUIP.HaloRegionAttr<shape = [1, 16, 4, 1], offset = [0, 0, 0, 0], cluster_id = 1 : i64>
    // CHECK:       ],
    // CHECK:       outwardHaloRegions = [
    // CHECK:           #VPUIP.OutwardHaloRegionAttr<shape = [1, 16, 4, 1], offset = [0, 16, 0, 0], cluster_id = 1 : i64, inwardHaloRegions = [
    // CHECK:               #VPUIP.HaloRegionAttr<shape = [1, 16, 4, 1], offset = [0, 16, 0, 0], cluster_id = 0 : i64>
    // CHECK:           ]>
    // CHECK:   ]>

    // CHECK:   [[OUT_TILE_1_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [1] <1856> -> !VPUIP.ITIBuffer<
    // CHECK:       1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 1],

    %WAIT_BARRIER = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[WAIT_BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%WAIT_BARRIER : !VPURT.Barrier) {
        %MATMUL = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            kernel_padding = #VPU.Padding<
                left = 0 : i64,
                right = 0 : i64,
                top = 0 : i64,
                bottom = 0 : i64
            >,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }>
        input(%IN : memref<2x1x16x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        weights(%WEIGHTS : memref<2x32x16x1x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        weight_table(%WEIGHT_TABLE : memref<2x32x1x1x4xsi32, [@CMX_NN, 0]>)
        parent_input(%IN : memref<2x1x16x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        parent_output(%OUT_TILE_0 : !VPUIP.ITIBuffer<
            2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0],
            inwardHaloRegions = [
                #VPUIP.HaloRegionAttr<shape = [2, 1, 16, 4, 1], offset = [0, 0, 16, 0, 0], cluster_id = 0 : i64>
            ],
            outwardHaloRegions = [
                #VPUIP.OutwardHaloRegionAttr<shape = [2, 1, 16, 4, 1], offset = [0, 0, 0, 0, 0], cluster_id = 0 : i64, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [2, 1, 16, 4, 1], offset = [0, 0, 0, 0, 0], cluster_id = 1 : i64>
                ]>
            ]>)
        output_ITI_buff(%OUT_TILE_1 : !VPUIP.ITIBuffer<
            2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 1],
            inwardHaloRegions = [
                #VPUIP.HaloRegionAttr<shape = [2, 1, 16, 4, 1], offset = [0, 0, 0, 0, 0], cluster_id = 1 : i64>
            ],
            outwardHaloRegions = [
                #VPUIP.OutwardHaloRegionAttr<shape = [2, 1, 16, 4, 1], offset = [0, 0, 16, 0, 0], cluster_id = 1 : i64, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [2, 1, 16, 4, 1], offset = [0, 0, 16, 0, 0], cluster_id = 0 : i64>
                ]>
            ]>)
        outputs(%OUT_TILE_0 : !VPUIP.ITIBuffer<
            2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0],
            inwardHaloRegions = [
                #VPUIP.HaloRegionAttr<shape = [2, 1, 16, 4, 1], offset = [0, 0, 16, 0, 0], cluster_id = 0 : i64>
            ],
            outwardHaloRegions = [
                #VPUIP.OutwardHaloRegionAttr<shape = [2, 1, 16, 4, 1], offset = [0, 0, 0, 0, 0], cluster_id = 0 : i64, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [2, 1, 16, 4, 1], offset = [0, 0, 0, 0, 0], cluster_id = 1 : i64>
                ]>
            ]>)
            -> !VPUIP.ITIBuffer<
            2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0],
            inwardHaloRegions = [
                #VPUIP.HaloRegionAttr<shape = [2, 1, 16, 4, 1], offset = [0, 0, 16, 0, 0], cluster_id = 0 : i64>
            ],
            outwardHaloRegions = [
                #VPUIP.OutwardHaloRegionAttr<shape = [2, 1, 16, 4, 1], offset = [0, 0, 0, 0, 0], cluster_id = 0 : i64, inwardHaloRegions = [
                    #VPUIP.HaloRegionAttr<shape = [2, 1, 16, 4, 1], offset = [0, 0, 0, 0, 0], cluster_id = 1 : i64>
                ]>
            ]>
        variants : {
            DPUTask {
                cluster_id = 0 : i64,
                inEnd = [0, 3, 15],
                inStart = [0, 0, 0],
                mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                outEnd = [0, 3, 15],
                outStart = [0, 0, 0],
                pad = #VPU.Padding<
                    left = 0 : i64,
                    right = 0 : i64,
                    top = 0 : i64,
                    bottom = 0 : i64
                >
            }
        } PPE : {
            PPETask {
                ppe = #VPU.PPEStub<>
            }
        }
    }

    // CHECK:       VPURT.Task updates([[WAIT_BARRIER]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[IN_SLICE_0]] : memref<1x16x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:           parent_output([[OUT_TILE_0_SLICE_0]]
    // CHECK:           output_ITI_buff([[OUT_TILE_1_SLICE_0]]
    // CHECK:           outputs([[OUT_TILE_0_SLICE_0]]

    // CHECK:       VPURT.Task updates([[WAIT_BARRIER]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[IN_SLICE_1]] : memref<1x16x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK:           parent_output([[OUT_TILE_0_SLICE_1]]
    // CHECK:           output_ITI_buff([[OUT_TILE_1_SLICE_1]]
    // CHECK:           outputs([[OUT_TILE_0_SLICE_1]]

    return
}

}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// Below test checks that unrolling pass didn't alias BufferDeclarations between 2 independent tasks.
// CHECK-LABEL: @UnrollingAvoidsBufferAliasing
module @UnrollingAvoidsBufferAliasing {

config.Resources 4 of @NCE at 1.850000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

func.func @main() -> () {
    %IN_A = VPURT.DeclareBuffer <CMX_NN> [0] <2112> -> memref<2x1x16x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    %WEIGHTS_A = VPURT.DeclareBuffer <CMX_NN> [0] <12864> -> memref<2x32x16x1x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    %OUT_A = VPURT.DeclareBuffer <CMX_NN> [0] <1600> -> memref<2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    %WEIGHT_TABLE_A = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<2x32x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[OUT_A_SLICE_1856:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1856> -> memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %WAIT_BARRIER_A = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[WAIT_BARRIER_A:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%WAIT_BARRIER_A : !VPURT.Barrier) {
        %MATMUL_A = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            kernel_padding = #VPU.Padding<
                left = 0 : i64,
                right = 0 : i64,
                top = 0 : i64,
                bottom = 0 : i64
            >,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }>
        input(%IN_A : memref<2x1x16x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        weights(%WEIGHTS_A : memref<2x32x16x1x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        weight_table(%WEIGHT_TABLE_A : memref<2x32x1x1x4xsi32, [@CMX_NN, 0]>)
        parent_input(%IN_A : memref<2x1x16x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        parent_output(%OUT_A : memref<2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        outputs(%OUT_A : memref<2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
            -> memref<2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
        variants : {
            DPUTask {
                cluster_id = 0 : i64,
                inEnd = [0, 3, 15],
                inStart = [0, 0, 0],
                mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                outEnd = [0, 3, 31],
                outStart = [0, 0, 0],
                pad = #VPU.Padding<
                    left = 0 : i64,
                    right = 0 : i64,
                    top = 0 : i64,
                    bottom = 0 : i64
                >
            }
        } PPE : {
            PPETask {
                ppe = #VPU.PPEStub<>
            }
        }
    }

    %IN_B = VPURT.DeclareBuffer <CMX_NN> [0] <20000> -> memref<2x1x16x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    %WEIGHTS_B = VPURT.DeclareBuffer <CMX_NN> [0] <30000> -> memref<2x32x16x1x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    %OUT_B = VPURT.DeclareBuffer <CMX_NN> [0] <1856> -> memref<2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    %WEIGHT_TABLE_B = VPURT.DeclareBuffer <CMX_NN> [0] <40000> -> memref<2x32x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[OUT_B_SLICE_1856:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1856> -> memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>

    %WAIT_BARRIER_B = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[WAIT_BARRIER_B:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    VPURT.Task updates(%WAIT_BARRIER_B : !VPURT.Barrier) {
        %MATMUL_B = VPUIP.NCEClusterTask <{resultSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            kernel_padding = #VPU.Padding<
                left = 0 : i64,
                right = 0 : i64,
                top = 0 : i64,
                bottom = 0 : i64
            >,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }>
        input(%IN_B : memref<2x1x16x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        weights(%WEIGHTS_B : memref<2x32x16x1x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        weight_table(%WEIGHT_TABLE_B : memref<2x32x1x1x4xsi32, [@CMX_NN, 0]>)
        parent_input(%IN_B : memref<2x1x16x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        parent_output(%OUT_B : memref<2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
        outputs(%OUT_B : memref<2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
            -> memref<2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
        variants : {
            DPUTask {
                cluster_id = 0 : i64,
                inEnd = [0, 3, 15],
                inStart = [0, 0, 0],
                mpe_mode = #VPU.mpe_mode<CUBOID_16x16>,
                outEnd = [0, 3, 31],
                outStart = [0, 0, 0],
                pad = #VPU.Padding<
                    left = 0 : i64,
                    right = 0 : i64,
                    top = 0 : i64,
                    bottom = 0 : i64
                >
            }
        } PPE : {
            PPETask {
                ppe = #VPU.PPEStub<>
            }
        }
    }

    // CHECK-DAG:   VPUIP.NCEClusterTask
    // CHECK-DAG-SAME:  outputs([[OUT_A_SLICE_1856]]
    // CHECK-DAG:   VPUIP.NCEClusterTask
    // CHECK-DAG-SAME:  outputs([[OUT_B_SLICE_1856]]

    return
}

}
