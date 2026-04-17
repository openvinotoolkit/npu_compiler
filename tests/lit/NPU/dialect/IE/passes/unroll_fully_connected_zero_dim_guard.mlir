//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --unroll-fully-connected %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @DontUnrollZeroBatchDim
// CHECK-SAME:   [[LHS:%arg[0-9]+]]: tensor<0x3072xf32>,
// CHECK-SAME:   [[WEIGHTS:%arg[0-9]+]]: tensor<1x1024x4096xf32>,
// CHECK-SAME:   [[IN_PARAM:%arg[0-9]+]]: tensor<1x1x1xf32>,
// CHECK-SAME:   [[OUT_PARAM:%arg[0-9]+]]: tensor<1x1x4096xf32>
func.func @DontUnrollZeroBatchDim(%LHS: tensor<0x3072xf32>,
                        %WEIGHTS: tensor<1x1024x4096xf32>,
                        %IN_PARAM: tensor<1x1x1xf32>,
                        %OUT_PARAM: tensor<1x1x4096xf32>) -> tensor<0x4096xf32> {
    %RHS_1 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x1024x4096xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x4096xf32>, tensor<1x1x4096xf32> -> tensor<1x1024x4096xf32>
    %RHS_2 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x1024x4096xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x4096xf32>, tensor<1x1x4096xf32> -> tensor<1x1024x4096xf32>
    %RHS_3 = IE.FakeQuantize(%WEIGHTS, %IN_PARAM, %IN_PARAM, %OUT_PARAM, %OUT_PARAM) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 16 : i64
    } : tensor<1x1024x4096xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x1x4096xf32>, tensor<1x1x4096xf32> -> tensor<1x1024x4096xf32>

    %CONCAT_RHS = IE.Concat(%RHS_1, %RHS_2, %RHS_3) {
        per_axis = #IE.Concat<axis = 0 : i64>
    } : tensor<1x1024x4096xf32>, tensor<1x1024x4096xf32>, tensor<1x1024x4096xf32> -> tensor<3x1024x4096xf32>

    %RESHAPE_RHS = IE.AffineReshape(%CONCAT_RHS) {
        dim_mapping = [[0], [0], [1]],
        shape_value = [3072, 4096]
    } : tensor<3x1024x4096xf32> -> tensor<3072x4096xf32>

    %TRANSPOSE_RHS = IE.Transpose(%RESHAPE_RHS) {
        order_value = #CN
    } : tensor<3072x4096xf32> -> tensor<4096x3072xf32>

    %GEMM = IE.FullyConnected(%LHS, %TRANSPOSE_RHS) : tensor<0x3072xf32>, tensor<4096x3072xf32> -> tensor<0x4096xf32>

    return %GEMM : tensor<0x4096xf32>

    // The zero batch dimension passes the existing inputChannels divisibility check
    // (3072 >= 3, 3072 % 3 == 0) but the defense-in-depth guard catches lhsShape[Dim(0)] <= 0
    // and returns failure, preserving the FC unchanged.
    // CHECK:   [[RHS_1:%.+]] = IE.FakeQuantize
    // CHECK:   [[RHS_2:%.+]] = IE.FakeQuantize
    // CHECK:   [[RHS_3:%.+]] = IE.FakeQuantize
    // CHECK:   [[CONCAT:%.+]] = IE.Concat([[RHS_1]], [[RHS_2]], [[RHS_3]])
    // CHECK:   [[RESHAPE:%.+]] = IE.AffineReshape([[CONCAT]])
    // CHECK:   [[TRANSPOSE:%.+]] = IE.Transpose([[RESHAPE]])
    // CHECK:   [[GEMM:%.+]] = IE.FullyConnected([[LHS]], [[TRANSPOSE]])

    // CHECK:   return [[GEMM]]
}
