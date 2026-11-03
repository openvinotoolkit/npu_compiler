//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --verify-diagnostics --init-compiler="platform=%platform%" --compute-nce-input-workloads --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: platform-NPU4000

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<!quant.quantile<ui4:f16, {0.000000e+00,1.000000e+00,2.000000e+00,3.000000e+00,4.000000e+00,5.000000e+00,6.000000e+00,7.000000e+00,-8.000000e+00,-7.000000e+00,-6.000000e+00,-5.000000e+00,-4.000000e+00,-3.000000e+00,-2.000000e+00,-1.000000e+00}>:f16, 1.000000e+00>

// CHECK: [[$Q_ELEM_TYPE:![a-zA-Z0-9_]+]] = !quant.uniform<!quant.quantile<ui4:f16,
// CHECK-LABEL: func.func @compressed_conv_palletized
// CHECK: [[WEIGHTS:%.+]] = VPUIP.ShapeCast {shape = [16, 32, 1, 1]}
// CHECK: [[INPUT:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 64, 4]}
func.func @compressed_conv_palletized(
        %input: tensor<1x4x64x4xf16, {order = #NHWC, mem_space = @CMX_NN}>,
        %weights: tensor<16x1x1x32x!qElemType, {order = #NHWC, mem_space = @CMX_NN}>,
        %weight_table: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}>)
        -> tensor<1x16x64x4xf16, {order = #NHWC, mem_space = @CMX_NN}> {
    %0 = VPU.NCE.CompressConvolution(%input, %weights, %weight_table) rawFilterShape [16, 4, 1, 1] {
        cm_sp_pattern = 15 : i64,
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        strides = [1, 1]
    } -> tensor<1x16x64x4xf16, {order = #NHWC, mem_space = @CMX_NN}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 64, 4] pad [0, 0, 0, 0] <CUBOID_16x16>
    }
    return %0 : tensor<1x16x64x4xf16, {order = #NHWC, mem_space = @CMX_NN}>
}

// CHECK: VPUIP.NCEClusterTask
// CHECK-SAME: input_channels_compression
// CHECK-SAME: input([[INPUT]] : memref<1x16x64x4xf16,
// CHECK-SAME: weights([[WEIGHTS]] : memref<16x32x1x1x[[$Q_ELEM_TYPE]],
// CHECK: DPUTask
// CHECK-SAME: inEnd = [3, 63, 15]
// CHECK-SAME: inStart = [0, 0, 0]
// CHECK: return
// CHECK-SAME: memref<1x16x64x4xf16,
