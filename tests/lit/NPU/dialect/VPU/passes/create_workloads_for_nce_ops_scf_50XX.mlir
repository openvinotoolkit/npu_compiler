//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --create-workloads-for-nce-ops-scf %s | FileCheck %s
// REQUIRES: platform-NPU5010

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#GNCHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

// NCE.MatMul 5D inside scf.forall — verifies 5D workload path produces a single
// workload with the full 5D shape and cluster_id = 0.
// CHECK-LABEL: @MatMulInsideForall
func.func @MatMulInsideForall(
        %arg0: tensor<1x1x16x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>,
        %weights: tensor<1x32x16x1x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>)
        -> tensor<6x1x32x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}> {

    %out = tensor.empty() : tensor<6x1x32x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>

    %result = scf.forall (%gid) = (0) to (6) step (1)
        shared_outs(%o = %out) -> (tensor<6x1x32x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>) {

        %mm = VPU.NCE.MatMul(%arg0, %weights) rawFilterShape [1, 32, 16, 1, 1] {
            resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
            strides = [1, 1]
        } : tensor<1x1x16x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>, tensor<1x32x16x1x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>
          -> tensor<1x1x32x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>

        scf.forall.in_parallel {
            tensor.parallel_insert_slice %mm into %o[%gid, 0, 0, 0, 0] [1, 1, 32, 4, 1] [1, 1, 1, 1, 1]
                : tensor<1x1x32x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>
                into tensor<6x1x32x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>
        }
    }

    return %result : tensor<6x1x32x4x1xf16, {mem_space = @CMX_NN, order = #GNHWC}>

    // CHECK:       scf.forall
    // CHECK:         VPU.NCE.MatMul
    // CHECK:           VPU.DPU.Workload inOffsets [0, 0, 0, 0, 0] inSizes [1, 1, 16, 4, 1] outOffsets [0, 0, 0, 0, 0] outSizes [1, 1, 32, 4, 1] pad [0, 0, 0, 0] <CUBOID_4x16> attributes {cluster_id = 0 : i64}
}
