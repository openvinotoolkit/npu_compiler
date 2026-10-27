//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --init-compiler="platform=%platform% allow-custom-values=true" --split-NCE-ops-onto-workloads %s | FileCheck %s
// REQUIRES: platform-NPU5010

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

// CHECK-LABEL: @SplitNCEMatMul
func.func @SplitNCEMatMul(
        %input: tensor<8x1x64x16x4xf16, {order = #GNHWC}>,
        %weights: tensor<8x256x64x1x1xf16, {order = #GNHWC}>,
        %weightsTable: tensor<8x256x1x1x4xsi32>)
        -> tensor<8x1x256x16x4xf16, {order = #GNHWC}> {
    %0 = VPU.NCE.MatMul(%input, %weights, %weightsTable) rawFilterShape [8, 256, 64, 1, 1] {
        resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        strides = [1, 1]
    } -> tensor<8x1x256x16x4xf16, {order = #GNHWC}>

    return %0 : tensor<8x1x256x16x4xf16, {order = #GNHWC}>

    // CHECK: VPU.NCE.MatMul
    // CHECK: VPU.DPU.Workload outOffsets [0, 0, 0, 0, 0] outSizes [8, 1, 256, 16, 4] pad [0, 0, 0, 0] <CUBOID_4x16>
}
