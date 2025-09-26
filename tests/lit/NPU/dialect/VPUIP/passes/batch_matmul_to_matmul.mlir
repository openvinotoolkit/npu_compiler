//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --batch-matmul-to-matmul %s | FileCheck %s
// REQUIRES: arch-NPU40XX

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @BatchMatMulToMatMul
module @BatchMatMulToMatMul attributes {
    config.arch = #config.arch_kind<NPU40XX>
} {

config.Resources 4 of @NCE at 1.850000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

func.func @main() -> () {
    %IN = VPURT.DeclareBuffer <CMX_NN> [0] <2112> -> memref<2x1x16x4x1xf16, #GNHWC, [@CMX_NN, 0]>
    // CHECK:   [[IN_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2112> -> memref<1x16x4x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[IN_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2240> -> memref<1x16x4x1xf16, #NHWC, [@CMX_NN, 0]>
    %WEIGHTS = VPURT.DeclareBuffer <CMX_NN> [0] <12864> -> memref<2x32x16x1x1xf16, #GNHWC, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHTS_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <12864> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHTS_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <13888> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %OUT = VPURT.DeclareBuffer <CMX_NN> [0] <1600> -> memref<2x1x32x4x1xf16, #GNHWC, [@CMX_NN, 0]>
    // CHECK:   [[OUT_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1600> -> memref<1x32x4x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[OUT_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1856> -> memref<1x32x4x1xf16, #NHWC, [@CMX_NN, 0]>
    %WEIGHT_TABLE = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<2x32x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHT_TABLE_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHT_TABLE_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1088> -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    %WAIT_BARRIER = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[WAIT_BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task updates(%WAIT_BARRIER : !VPURT.Barrier) {
        %MATMUL = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<
                left = 0 : i64,
                right = 0 : i64,
                top = 0 : i64,
                bottom = 0 : i64
            >,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%IN : memref<2x1x16x4x1xf16, #GNHWC, [@CMX_NN, 0]>)
        weights(%WEIGHTS : memref<2x32x16x1x1xf16, #GNHWC, [@CMX_NN, 0]>)
        weight_table(%WEIGHT_TABLE : memref<2x32x1x1x4xsi32, [@CMX_NN, 0]>)
        parent_input(%IN : memref<2x1x16x4x1xf16, #GNHWC, [@CMX_NN, 0]>)
        parent_output(%OUT : memref<2x1x32x4x1xf16, #GNHWC, [@CMX_NN, 0]>)
        outputs(%OUT : memref<2x1x32x4x1xf16, #GNHWC, [@CMX_NN, 0]>)
            -> memref<2x1x32x4x1xf16, #GNHWC, [@CMX_NN, 0]>
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
    // CHECK-SAME:      input([[IN_SLICE_0]] : memref<1x16x4x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      weights([[WEIGHTS_SLICE_0]] : memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      weight_table([[WEIGHT_TABLE_SLICE_0]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_input([[IN_SLICE_0]] : memref<1x16x4x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_output([[OUT_SLICE_0]] : memref<1x32x4x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[OUT_SLICE_0]] : memref<1x32x4x1xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       VPURT.Task updates([[WAIT_BARRIER]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[IN_SLICE_1]] : memref<1x16x4x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      weights([[WEIGHTS_SLICE_1]] : memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      weight_table([[WEIGHT_TABLE_SLICE_1]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_input([[IN_SLICE_1]] : memref<1x16x4x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_output([[OUT_SLICE_1]] : memref<1x32x4x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[OUT_SLICE_1]] : memref<1x32x4x1xf16, #NHWC, [@CMX_NN, 0]>)

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
module @QuantBatchMatMulToMatMul attributes {
    config.arch = #config.arch_kind<NPU40XX>
} {

config.Resources 4 of @NCE at 1.850000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

func.func @main() -> () {
    %IN = VPURT.DeclareBuffer <CMX_NN> [0] <2112> -> memref<2x1x16x4x1x!qElemType, #GNHWC, [@CMX_NN, 0]>
    // CHECK:   [[IN_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2112> -> memref<1x16x4x1x[[Q_ELEM_TYPE]], #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[IN_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2176> -> memref<1x16x4x1x[[Q_ELEM_TYPE]], #NHWC, [@CMX_NN, 0]>
    %WEIGHT = VPURT.DeclareBuffer <CMX_NN> [0] <12864> -> memref<2x32x16x1x1xf16, #GNHWC, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHT_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <12864> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHT_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <13888> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %OUT = VPURT.DeclareBuffer <CMX_NN> [0] <1600> -> memref<2x1x32x4x1xf16, #GNHWC, [@CMX_NN, 0]>
    // CHECK:   [[OUT_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1600> -> memref<1x32x4x1xf16, #NHWC, [@CMX_NN, 0]>
    // CHECK:   [[OUT_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1856> -> memref<1x32x4x1xf16, #NHWC, [@CMX_NN, 0]>
    %WEIGHT_TABLE = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<2x32x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHT_TABLE_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHT_TABLE_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1088> -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    %WAIT_BARRIER = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[WAIT_BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task updates(%WAIT_BARRIER : !VPURT.Barrier) {
        %MATMUL = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<
                left = 0 : i64,
                right = 0 : i64,
                top = 0 : i64,
                bottom = 0 : i64
            >,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%IN : memref<2x1x16x4x1x!qElemType, #GNHWC, [@CMX_NN, 0]>)
        weights(%WEIGHT : memref<2x32x16x1x1xf16, #GNHWC, [@CMX_NN, 0]>)
        weight_table(%WEIGHT_TABLE : memref<2x32x1x1x4xsi32, [@CMX_NN, 0]>)
        parent_input(%IN : memref<2x1x16x4x1x!qElemType, #GNHWC, [@CMX_NN, 0]>)
        parent_output(%OUT : memref<2x1x32x4x1xf16, #GNHWC, [@CMX_NN, 0]>)
        outputs(%OUT : memref<2x1x32x4x1xf16, #GNHWC, [@CMX_NN, 0]>)
            -> memref<2x1x32x4x1xf16, #GNHWC, [@CMX_NN, 0]>
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
    // CHECK-SAME:      input([[IN_SLICE_0]] : memref<1x16x4x1x[[Q_ELEM_TYPE]], #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      weights([[WEIGHTS_SLICE_0]] : memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      weight_table([[WEIGHT_TABLE_SLICE_0]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_input([[IN_SLICE_0]] : memref<1x16x4x1x[[Q_ELEM_TYPE]], #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_output([[OUT_SLICE_0]] : memref<1x32x4x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[OUT_SLICE_0]] : memref<1x32x4x1xf16, #NHWC, [@CMX_NN, 0]>)

    // CHECK:       VPURT.Task updates([[WAIT_BARRIER]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[IN_SLICE_1]] : memref<1x16x4x1x[[Q_ELEM_TYPE]], #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      weights([[WEIGHTS_SLICE_1]] : memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      weight_table([[WEIGHT_TABLE_SLICE_1]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_input([[IN_SLICE_1]] : memref<1x16x4x1x[[Q_ELEM_TYPE]], #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_output([[OUT_SLICE_1]] : memref<1x32x4x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[OUT_SLICE_1]] : memref<1x32x4x1xf16, #NHWC, [@CMX_NN, 0]>)

    return
}

}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @MatMulWithBatch1ToMatMul
module @MatMulWithBatch1ToMatMul attributes {
    config.arch = #config.arch_kind<NPU40XX>
} {

config.Resources 4 of @NCE at 1.850000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

func.func @main() -> () {
    %IN = VPURT.DeclareBuffer <CMX_NN> [0] <2112> -> memref<1x1x16x4x1xf16, #GNHWC, [@CMX_NN, 0]>
    // CHECK:   [[IN:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2112> -> memref<1x16x4x1xf16, #NHWC, [@CMX_NN, 0]>
    %WEIGHTS = VPURT.DeclareBuffer <CMX_NN> [0] <12864> -> memref<1x32x16x1x1xf16, #GNHWC, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHTS:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <12864> -> memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %OUT = VPURT.DeclareBuffer <CMX_NN> [0] <1600> -> memref<1x1x32x4x1xf16, #GNHWC, [@CMX_NN, 0]>
    // CHECK:   [[OUT:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1600> -> memref<1x32x4x1xf16, #NHWC, [@CMX_NN, 0]>
    %WEIGHT_TABLE = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<1x32x1x1x4xsi32, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHT_TABLE:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <576> -> memref<32x1x1x4xsi32, [@CMX_NN, 0]>
    %WAIT_BARRIER = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    // CHECK:   [[WAIT_BARRIER:%.+]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    VPURT.Task updates(%WAIT_BARRIER : !VPURT.Barrier) {
        %MATMUL = VPUIP.NCEClusterTask {
            kernel_padding = #VPU.Padding<
                left = 0 : i64,
                right = 0 : i64,
                top = 0 : i64,
                bottom = 0 : i64
            >,
            kernel_size = [1, 1],
            kernel_strides = [1, 1],
            task_type = #VPUIP.nce_task_type<CONV>
        }
        input(%IN : memref<1x1x16x4x1xf16, #GNHWC, [@CMX_NN, 0]>)
        weights(%WEIGHTS : memref<1x32x16x1x1xf16, #GNHWC, [@CMX_NN, 0]>)
        weight_table(%WEIGHT_TABLE : memref<1x32x1x1x4xsi32, [@CMX_NN, 0]>)
        parent_input(%IN : memref<1x1x16x4x1xf16, #GNHWC, [@CMX_NN, 0]>)
        parent_output(%OUT : memref<1x1x32x4x1xf16, #GNHWC, [@CMX_NN, 0]>)
        outputs(%OUT : memref<1x1x32x4x1xf16, #GNHWC, [@CMX_NN, 0]>)
            -> memref<1x1x32x4x1xf16, #GNHWC, [@CMX_NN, 0]>
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
    // CHECK-SAME:      input([[IN]] : memref<1x16x4x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      weights([[WEIGHTS]] : memref<32x16x1x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      weight_table([[WEIGHT_TABLE]] : memref<32x1x1x4xsi32, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_input([[IN]] : memref<1x16x4x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_output([[OUT]] : memref<1x32x4x1xf16, #NHWC, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[OUT]] : memref<1x32x4x1xf16, #NHWC, [@CMX_NN, 0]>)

    return
}

}
