//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --batch-matmul-to-matmul %s | FileCheck %s
// REQUIRES: platform-NPU6010

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

!qElemType = !quant.uniform<i8:f16:1, {1.0:1,1.0:2,1.0:3,1.0:4,1.0:5,1.0:6,1.0:7,1.0:8,1.0:9,1.0:10,1.0:11,1.0:12,1.0:13,1.0:14,1.0:15,1.0:16,1.0:17,1.0:18,1.0:19,1.0:20,1.0:21,1.0:22,1.0:23,1.0:24,1.0:25,1.0:26,1.0:27,1.0:28,1.0:29,1.0:30,1.0:31,1.0:32}>

// CHECK: !qElemType = !quant.uniform<i8:f16:0, {1.000000e+00:1,1.000000e+00:2,1.000000e+00:3,1.000000e+00:4,1.000000e+00:5,1.000000e+00:6,1.000000e+00:7,1.000000e+00:8,1.000000e+00:9,1.000000e+00:10,1.000000e+00:11,1.000000e+00:12,1.000000e+00:13,1.000000e+00:14,1.000000e+00:15,1.000000e+00:16,1.000000e+00:17,1.000000e+00:18,1.000000e+00:19,1.000000e+00:20,1.000000e+00:21,1.000000e+00:22,1.000000e+00:23,1.000000e+00:24,1.000000e+00:25,1.000000e+00:26,1.000000e+00:27,1.000000e+00:28,1.000000e+00:29,1.000000e+00:30,1.000000e+00:31,1.000000e+00:32}>

// CHECK-LABEL: @BatchMatMulToMatMul
module @BatchMatMulToMatMul {

config.Resources 4 of @NCE at 1.850000e+03 MHz {
    config.ExecutorResource 1 of @DPU
}

func.func @main() -> () {
    %IN = VPURT.DeclareBuffer <CMX_NN> [0] <2112> -> memref<2x1x16x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    // CHECK:   [[IN_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2112> -> memref<1x16x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[IN_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2240> -> memref<1x16x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %WEIGHTS = VPURT.DeclareBuffer <CMX_NN> [0] <2368> -> memref<2x32x16x1x1x!qElemType, {order = #GNHWC}, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHTS_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2368> -> memref<32x16x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[WEIGHTS_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <2880> -> memref<32x16x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>
    %OUT = VPURT.DeclareBuffer <CMX_NN> [0] <1600> -> memref<2x1x32x4x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_SLICE_0:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1600> -> memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    // CHECK:   [[OUT_SLICE_1:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1856> -> memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>
    %SCALE_TABLE = VPURT.DeclareBuffer <CMX_NN> [0] <1088> -> memref<1x32x1x1x1xf32, [@CMX_NN, 0]>
    // CHECK:   [[SCALE_TABLE:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1088> -> memref<32x1x1x1xf32, [@CMX_NN, 0]>
    %BIAS_TABLE = VPURT.DeclareBuffer <CMX_NN> [0] <1344> -> memref<1x32x1x1x1xf32, [@CMX_NN, 0]>
    // CHECK:   [[BIAS_TABLE:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <1344> -> memref<32x1x1x1xf32, [@CMX_NN, 0]>
    %ZERO_POINT_TABLE = VPURT.DeclareBuffer <CMX_NN> [0] <960> -> memref<1x32x1x1x1xi8, [@CMX_NN, 0]>
    // CHECK:   [[ZERO_POINT_TABLE:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <960> -> memref<32x1x1x1xi8, [@CMX_NN, 0]>
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
        weights(%WEIGHTS : memref<2x32x16x1x1x!qElemType, {order = #GNHWC}, [@CMX_NN, 0]>)
        weight_table_scale(%SCALE_TABLE : memref<1x32x1x1x1xf32, [@CMX_NN, 0]>)
        weight_table_bias(%BIAS_TABLE : memref<1x32x1x1x1xf32, [@CMX_NN, 0]>)
        weight_zero_points(%ZERO_POINT_TABLE : memref<1x32x1x1x1xi8, [@CMX_NN, 0]>)
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
    // CHECK-SAME:      weights([[WEIGHTS_SLICE_0]] : memref<32x16x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      weight_table_scale([[SCALE_TABLE]] : memref<32x1x1x1xf32, [@CMX_NN, 0]>)
    // CHECK-SAME:      weight_table_bias([[BIAS_TABLE]] : memref<32x1x1x1xf32, [@CMX_NN, 0]>)
    // CHECK-SAME:      weight_zero_points([[ZERO_POINT_TABLE]] : memref<32x1x1x1xi8, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_input([[IN_SLICE_0]] : memref<1x16x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_output([[OUT_SLICE_0]] : memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[OUT_SLICE_0]] : memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    // CHECK:       VPURT.Task updates([[WAIT_BARRIER]] : !VPURT.Barrier) {
    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      input([[IN_SLICE_1]] : memref<1x16x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      weights([[WEIGHTS_SLICE_1]] : memref<32x16x1x1x!qElemType, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      weight_table_scale([[SCALE_TABLE]] : memref<32x1x1x1xf32, [@CMX_NN, 0]>)
    // CHECK-SAME:      weight_table_bias([[BIAS_TABLE]] : memref<32x1x1x1xf32, [@CMX_NN, 0]>)
    // CHECK-SAME:      weight_zero_points([[ZERO_POINT_TABLE]] : memref<32x1x1x1xi8, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_input([[IN_SLICE_1]] : memref<1x16x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      parent_output([[OUT_SLICE_1]] : memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:      outputs([[OUT_SLICE_1]] : memref<1x32x4x1xf16, {order = #NHWC}, [@CMX_NN, 0]>)

    return
}

}
