//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --one-shot-bufferize-VPU-to-VPUIP --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.19364075006223191:128>

// CHECK-LABEL: @Int8ActDepthConvWithoutL1Opt
// CHECK-SAME: [[INPUT:%.+]]: memref<1x80x64x64x!qElemType, {order = #NHWC}, @CMX_NN>
// CHECK-SAME: [[WEIGHTS:%.+]]: memref<80x16x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
// CHECK-SAME: [[WEIGHT_TABLE:%.+]]: memref<80x1x1x4xsi32, @CMX_NN>
// CHECK-SAME: -> memref<1x80x64x64x!qElemType, {order = #NHWC}, @CMX_NN>
func.func @Int8ActDepthConvWithoutL1Opt(
    %input: tensor<1x80x64x64x!qElemType, {order = #NHWC, mem_space = @CMX_NN}>,
    %weights: tensor<80x16x1x1x!qElemType, {order = #NHWC, mem_space = @CMX_NN}>,
    %weight_table: tensor<80x1x1x4xsi32, {mem_space = @CMX_NN}>
) -> tensor<1x80x64x64x!qElemType, {order = #NHWC, mem_space = @CMX_NN}> {

    %0 = VPU.NCE.DepthConvolution(%input, %weights, %weight_table) rawFilterShape [80, 1, 3, 3] {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.280000e+02 : f64, clamp_high = 1.270000e+02 : f64,
            scale = 0.073115045214323729 : f64, prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64, adder = 1.280000e+02 : f64>,

        strides = [1, 1]
    } -> tensor<1x80x64x64x!qElemType, {order = #NHWC, mem_space = @CMX_NN}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 64, 64] pad [1, 1, 1, 1] #VPU.mpe_mode<CUBOID_16x16>
        VPU.DPU.Workload outOffsets [0, 16, 0, 0] outSizes [1, 64, 64, 64] pad [1, 1, 1, 1] #VPU.mpe_mode<CUBOID_16x16>
    }

    return %0 : tensor<1x80x64x64x!qElemType, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x80x64x64x!qElemType, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[RES:%.+]] = VPUIP.NCEClusterTask
    // CHECK-NOT:       is_small_kernel_optimized
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<DWCONV>
    // CHECK-SAME:  input([[INPUT]] : memref<1x80x64x64x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  weights([[WEIGHTS]] : memref<80x16x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  weight_table([[WEIGHT_TABLE]] : memref<80x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:  parent_input([[INPUT]] : memref<1x80x64x64x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<1x80x64x64x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUT_BUF]] : memref<1x80x64x64x!qElemType, {order = #NHWC}, @CMX_NN>)

    // CHECK: return [[RES]] : memref<1x80x64x64x!qElemType, {order = #NHWC}, @CMX_NN>
}
