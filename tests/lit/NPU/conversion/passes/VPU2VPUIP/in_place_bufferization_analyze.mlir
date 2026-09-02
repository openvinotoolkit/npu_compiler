//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --in-place-bufferization-analyze %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NceEltwiseAdd
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME: [[ARG1:%.+]]: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
// CHECK-SAME: -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
func.func @NceEltwiseAdd(%arg0: tensor<1x64x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}>,
                         %arg1: tensor<1x64x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}> {

    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                op_type = #VPU.eltwise_type<ADD>,
                ppe = #VPU.PPEStub<>
            } -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 28, 28] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
    }

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK: [[ELTWISE_ADD:%.+]] = VPU.NCE.Eltwise([[ARG0]], [[ARG1]]) {
    // CHECK-SAME: __inplace_operands_attr__ = ["false", "false"],
    // CHECK-SAME: op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME: ppe = #VPU.PPEStub<>, resultSegmentSizes = array<i32: 1, 0, 0, 0>}
    // CHECK-SAME: -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    // CHECK: VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 28, 28] pad [0, 0, 0, 0] <VECTOR_FP16>
    // CHECK: }

    // CHECK: return {__inplace_operands_attr__ = ["true"]} [[ELTWISE_ADD]] : tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>
}
