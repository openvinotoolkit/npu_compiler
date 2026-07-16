//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="platform=%platform%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NcePermute
// CHECK-SAME: (
// CHECK-SAME: [[ARG0:%.+]]: memref<1x3x224x224xf16, @CMX_NN>
// CHECK-SAME: )
// CHECK-SAME: -> memref<1x4x224x224xf16, {order = #NHWC}, @CMX_NN>
func.func @NcePermute(%arg0: tensor<1x3x224x224xf16, {mem_space = @CMX_NN}>)
        -> tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = f16,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [5.000000e-01], fp_prelu_alpha = 5.000000e-01 : f64>
    } -> tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 224, 224] pad [0, 0, 0, 0] <CUBOID_16x16>
    }

    return %0 : tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK: [[VIEW_OP_IN:%.+]] = VPUIP.ViewOp [[ARG0]] : memref<1x3x224x224xf16, @CMX_NN>
    // CHECK-SAME:  to memref<1x224x3x224xf16, {order = #NHWC}, @CMX_NN>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x224x4x224xf16, {order = #NWCH}, @CMX_NN>

    // CHECK:       [[PERMUTE_RES:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:  input([[VIEW_OP_IN]] : memref<1x224x3x224xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  weights([[VIEW_OP_IN]] : memref<1x224x3x224xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  parent_input([[VIEW_OP_IN]] : memref<1x224x3x224xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<1x224x4x224xf16, {order = #NWCH}, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUT_BUF]] : memref<1x224x4x224xf16, {order = #NWCH}, @CMX_NN>)
    // CHECK-SAME:  -> memref<1x224x4x224xf16, {order = #NWCH}, @CMX_NN>

    // CHECK:       PPETask {ppe = #VPU.PPEInt<
    // CHECK-SAME:      mode = <ADD>,
    // CHECK-SAME:      clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
    // CHECK-SAME:      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
    // CHECK-SAME:      quant_scale = [5.000000e-01],
    // CHECK-SAME:      fp_prelu_alpha = 5.000000e-01 : f64
    // CHECK-SAME:  >}

    // CHECK: [[VIEW_OP_OUT:%.+]] = VPUIP.ViewOp [[PERMUTE_RES]] : memref<1x224x4x224xf16, {order = #NWCH}, @CMX_NN>
    // CHECK-SAME: to memref<1x4x224x224xf16, {order = #NHWC}, @CMX_NN>

    // CHECK: return [[VIEW_OP_OUT]] : memref<1x4x224x224xf16, {order = #NHWC}, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @NcePermuteQuantOut
// CHECK-SAME: (
// CHECK-SAME: [[ARG0:%.+]]: memref<1x3x224x224xf16, @CMX_NN>
// CHECK-SAME: )
// CHECK-SAME: -> memref<1x4x224x224x!qElemType, {order = #NHWC}, @CMX_NN>
func.func @NcePermuteQuantOut(%arg0: tensor<1x3x224x224xf16, {mem_space = @CMX_NN}>)
        -> tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> {

    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [5.000000e-01], fp_prelu_alpha = 5.000000e-01 : f64>
    } -> tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 224, 224] pad [0, 0, 0, 0] <CUBOID_16x16>
    }

    return %0 : tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK: [[VIEW_OP_IN:%.+]] = VPUIP.ViewOp [[ARG0]] : memref<1x3x224x224xf16, @CMX_NN>
    // CHECK-SAME:  to memref<1x224x3x224xf16, {order = #NHWC}, @CMX_NN>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x224x4x224x!qElemType, {order = #NWCH}, @CMX_NN>

    // CHECK:       [[PERMUTE_RES:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:  input([[VIEW_OP_IN]] : memref<1x224x3x224xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  weights([[VIEW_OP_IN]] : memref<1x224x3x224xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  parent_input([[VIEW_OP_IN]] : memref<1x224x3x224xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<1x224x4x224x!qElemType, {order = #NWCH}, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUT_BUF]] : memref<1x224x4x224x!qElemType, {order = #NWCH}, @CMX_NN>)
    // CHECK-SAME:  -> memref<1x224x4x224x!qElemType, {order = #NWCH}, @CMX_NN>

    // CHECK:       PPETask {ppe = #VPU.PPEInt<
    // CHECK-SAME:      mode = <ADD>,
    // CHECK-SAME:      clamp_low = 0 : i64, clamp_high = 255 : i64,
    // CHECK-SAME:      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
    // CHECK-SAME:      quant_scale = [5.000000e-01],
    // CHECK-SAME:      fp_prelu_alpha = 5.000000e-01 : f64
    // CHECK-SAME:  >}

    // CHECK: [[VIEW_OP_OUT:%.+]] = VPUIP.ViewOp [[PERMUTE_RES]] : memref<1x224x4x224x!qElemType, {order = #NWCH}, @CMX_NN>
    // CHECK-SAME: to memref<1x4x224x224x!qElemType, {order = #NHWC}, @CMX_NN>

    // CHECK: return [[VIEW_OP_OUT]] : memref<1x4x224x224x!qElemType, {order = #NHWC}, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 0.0078431372549019607>

// CHECK-LABEL: @NcePermuteQuantInQuantOut
// CHECK-SAME: (
// CHECK-SAME: [[ARG0:%.+]]: memref<1x3x224x224x!qElemType, @CMX_NN>
// CHECK-SAME: )
// CHECK-SAME: -> memref<1x4x224x224x!qElemType, {order = #NHWC}, @CMX_NN>
func.func @NcePermuteQuantInQuantOut(%arg0: tensor<1x3x224x224x!qElemType, {mem_space = @CMX_NN}>)
        -> tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> {

    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_mult = [16384], quant_shift = [29], quant_post_shift = 0 : i64, in1_quant_mult = [16384], in2_quant_mult = [16384], fp_prelu_alpha = 1.000000e+00 : f64>
    } -> tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 224, 224] pad [0, 0, 0, 0] <CUBOID_16x16>
    }

    return %0 : tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK: [[VIEW_OP_IN:%.+]] = VPUIP.ViewOp [[ARG0]] : memref<1x3x224x224x!qElemType, @CMX_NN>
    // CHECK-SAME:  to memref<1x224x3x224x!qElemType, {order = #NHWC}, @CMX_NN>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x224x4x224x!qElemType, {order = #NWCH}, @CMX_NN>

    // CHECK:       [[PERMUTE_RES:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:  input([[VIEW_OP_IN]] : memref<1x224x3x224x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  weights([[VIEW_OP_IN]] : memref<1x224x3x224x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  parent_input([[VIEW_OP_IN]] : memref<1x224x3x224x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<1x224x4x224x!qElemType, {order = #NWCH}, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUT_BUF]] : memref<1x224x4x224x!qElemType, {order = #NWCH}, @CMX_NN>)
    // CHECK-SAME:  -> memref<1x224x4x224x!qElemType, {order = #NWCH}, @CMX_NN>

    // CHECK:       PPETask {ppe = #VPU.PPEInt<
    // CHECK-SAME:      mode = <ADD>,
    // CHECK-SAME:      clamp_low = 0 : i64, clamp_high = 255 : i64,
    // CHECK-SAME:      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64,
    // CHECK-SAME:      quant_mult = [16384], quant_shift = [29], quant_post_shift = 0 : i64,
    // CHECK-SAME:      in1_quant_mult = [16384], in2_quant_mult = [16384],
    // CHECK-SAME:      fp_prelu_alpha = 1.000000e+00 : f64
    // CHECK-SAME:  >}

    // CHECK: [[VIEW_OP_OUT:%.+]] = VPUIP.ViewOp [[PERMUTE_RES]] : memref<1x224x4x224x!qElemType, {order = #NWCH}, @CMX_NN>
    // CHECK-SAME: to memref<1x4x224x224x!qElemType, {order = #NHWC}, @CMX_NN>

    // CHECK: return [[VIEW_OP_OUT]] : memref<1x4x224x224x!qElemType, {order = #NHWC}, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

!InputTensor = !VPU.DistributedTensor<
    1x3x256x224xf16, #NCHW, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    strides = [2, 1],
    num_clusters = 2 : i64
}>

!OutputTensor = !VPU.DistributedTensor<
    1x4x256x224x!qElemType, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [1, 1],
    pads = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
    strides = [2, 1],
    num_clusters = 2,
    equal_memory_and_compute_view
}>

// CHECK-LABEL: @NcePermute_MultiTile
// CHECK-SAME: (
// CHECK-SAME: [[ARG0:%.+]]: !VPUIP.DistributedBuffer<1x3x256x224xf16, #NCHW, @CMX_NN,
// CHECK-SAME: )
// CHECK-SAME: -> !VPUIP.DistributedBuffer<1x4x256x224x!qElemType, #NHWC, @CMX_NN,
func.func @NcePermute_MultiTile(%in: !InputTensor) -> !OutputTensor {
    %out = VPU.NCE.Permute(%in) {
            dstElemType = !qElemType,
            dstOrder = #NHWC,
            expandedChannels = 4 : i64,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = 0 : i64, clamp_high = 255 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [5.000000e-01], fp_prelu_alpha = 5.000000e-01 : f64>
        } -> !OutputTensor {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 256, 224] pad [0, 0, 0, 0] <CUBOID_16x16>
        }

    return %out : !OutputTensor

    // CHECK:       [[VIEW_OP_IN:%.+]] = VPUIP.ViewOp [[ARG0]] :
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x3x256x224xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:    to
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x224x3x256xf16, #NHWC, @CMX_NN,

    // CHECK:       [[OUT_BUF:%.+]] = VPURT.AllocDistributed
    // CHECK-SAME:    -> !VPUIP.DistributedBuffer<1x224x4x256x!qElemType, #NWCH, @CMX_NN,
    // CHECK-SAME:   {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [1, 1],
    // CHECK-SAME:   pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:   strides = [1, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>

    // CHECK:        [[MULTI_TILE_RES:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:          is_permute_quantize
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:          input([[VIEW_OP_IN]] : !VPUIP.DistributedBuffer<1x224x3x256xf16, #NHWC, @CMX_NN
    // CHECK-SAME:          weights([[VIEW_OP_IN]] : !VPUIP.DistributedBuffer<1x224x3x256xf16, #NHWC, @CMX_NN
    // CHECK-SAME:          parent_input([[VIEW_OP_IN]] : !VPUIP.DistributedBuffer<1x224x3x256xf16, #NHWC, @CMX_NN
    // CHECK-SAME:          parent_output([[OUT_BUF]] : !VPUIP.DistributedBuffer<1x224x4x256x!qElemType, #NWCH, @CMX_NN
    // CHECK-SAME:          outputs([[OUT_BUF]] : !VPUIP.DistributedBuffer<1x224x4x256x!qElemType, #NWCH, @CMX_NN
    // CHECK-SAME:     -> !VPUIP.DistributedBuffer<1x224x4x256x!qElemType, #NWCH, @CMX_NN,
    // CHECK-SAME:         {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [1, 1],
    // CHECK-SAME:         pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:         strides = [1, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>

    // CHECK:       [[RES:%.+]] = VPUIP.ViewOp [[MULTI_TILE_RES]] :
    // CHECK-SAME:   !VPUIP.DistributedBuffer<1x224x4x256x!qElemType, #NWCH, @CMX_NN,
    // CHECK-SAME:   to
    // CHECK-SAME:   !VPUIP.DistributedBuffer<1x4x256x224x!qElemType, #NHWC, @CMX_NN,

    // CHECK: return [[RES]] : !VPUIP.DistributedBuffer<1x4x256x224x!qElemType, #NHWC, @CMX_NN,
}
