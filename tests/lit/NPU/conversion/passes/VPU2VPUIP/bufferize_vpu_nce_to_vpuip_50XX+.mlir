//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --one-shot-bufferize-VPU-to-VPUIP --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// NOTE: This test-case is needed in order to ensure functionality of errata E#94064 for VPUX50XX
// CHECK-LABEL: @ErrataSuperdenseNCEAveragePool
func.func @ErrataSuperdenseNCEAveragePool(%arg0: tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x16x15x13xf16, {mem_space = @CMX_NN, order = #NCHW}> {
    %1 = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 3],
        minimumHardwareExecutionCost = 708 : i64,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
        strides = [1, 1]
    } -> tensor<1x16x15x13xf16, {mem_space = @CMX_NN, order = #NCHW}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 7, 4] pad [1, 0, 0, 13] #VPU.mpe_mode<CUBOID_16x16>
    }

    return %1 : tensor<1x16x15x13xf16, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK:       VPUIP.NCEClusterTask {minimumHardwareExecutionCost = 708 : i64} <{
    // CHECK-NOT:       is_small_kernel_optimized,
    // CHECK-SAME:      is_superdense,
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<AVEPOOL>
    // CHECK-SAME:  }>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseSubtract
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x64x28x28xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME: [[ARG1:%.+]]: memref<1x64x28x28xf16, {order = #NHWC}, @CMX_NN>)
func.func @EltwiseSubtract(%arg0: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %1 = VPU.NCE.Eltwise(%arg0, %arg1) {
                op_type = #VPU.eltwise_type<SUBTRACT>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
            } -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 5, 28] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
        VPU.DPU.Workload outOffsets [0, 0, 5, 0] outSizes [1, 64, 5, 28] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
        VPU.DPU.Workload outOffsets [0, 0, 10, 0] outSizes [1, 64, 5, 28] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
        VPU.DPU.Workload outOffsets [0, 0, 15, 0] outSizes [1, 64, 5, 28] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
        VPU.DPU.Workload outOffsets [0, 0, 20, 0] outSizes [1, 64, 8, 28] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
    }
    return %1 : tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[ALLOC0:%.+]] = memref.alloc() : memref<1x64x28x28xf16, {order = #NHWC}, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK:               eltwise_type = #VPU.eltwise_type<SUBTRACT>
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:      input([[ARG0]] : memref<1x64x28x28xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      weights([[ARG1]] : memref<1x64x28x28xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      parent_input([[ARG0]] : memref<1x64x28x28xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      parent_output([[ALLOC0]] : memref<1x64x28x28xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:      outputs([[ALLOC0]] : memref<1x64x28x28xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK:           DPUTask
    // CHECK-SAME:          <VECTOR_FP16>,
    // CHECK-SAME:          outEnd = [27, 4, 63],
    // CHECK-SAME:          outStart = [0, 0, 0],
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:           DPUTask
    // CHECK-SAME:          <VECTOR_FP16>,
    // CHECK-SAME:          outEnd = [27, 9, 63],
    // CHECK-SAME:          outStart = [0, 5, 0],
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:           DPUTask
    // CHECK-SAME:          <VECTOR_FP16>,
    // CHECK-SAME:          outEnd = [27, 14, 63],
    // CHECK-SAME:          outStart = [0, 10, 0],
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:           DPUTask
    // CHECK-SAME:          <VECTOR_FP16>,
    // CHECK-SAME:          outEnd = [27, 19, 63],
    // CHECK-SAME:          outStart = [0, 15, 0],
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:           DPUTask
    // CHECK-SAME:          <VECTOR_FP16>,
    // CHECK-SAME:          outEnd = [27, 27, 63],
    // CHECK-SAME:          outStart = [0, 20, 0],
    // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK:    PPETask {ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NceEltwiseMultiply
// CHECK-SAME: (
// CHECK-SAME: [[ARG0:%.+]]: memref<1x64x28x28xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME: [[ARG1:%.+]]: memref<1x64x28x28xf16, {order = #NHWC}, @CMX_NN>
// CHECK-SAME: )
// CHECK-SAME: -> memref<1x64x28x28xf16, {order = #NHWC}, @CMX_NN>
func.func @NceEltwiseMultiply(%arg0: tensor<1x64x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}>,
                         %arg1: tensor<1x64x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}> {

    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
                op_type = #VPU.eltwise_type<MULTIPLY>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
            } -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 28, 28] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
    }

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x64x28x28xf16, {order = #NHWC}, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK:               eltwise_type = #VPU.eltwise_type<MULTIPLY>
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:  input([[ARG0]] : memref<1x64x28x28xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  weights([[ARG1]] : memref<1x64x28x28xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  parent_input([[ARG0]] : memref<1x64x28x28xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<1x64x28x28xf16, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUT_BUF]] : memref<1x64x28x28xf16, {order = #NHWC}, @CMX_NN>)

    // CHECK: return
    // CHECK-SAME: memref<1x64x28x28xf16, {order = #NHWC}, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEReduce
func.func @NCEReduce(%arg0: tensor<1x64x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}>)
        -> tensor<1x1x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}> {

    %0 = VPU.NCE.Reduce(%arg0) {
                axes = [1],
                op_type = #VPU.reduce_type<MEAN>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                    scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
            } -> tensor<1x1x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 28, 28] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_16x16>
    }

    return %0 : tensor<1x1x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK:       VPUIP.NCEClusterTask <{
    // CHECK-SAME:      is_superdense,
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<REDUCEMEAN>
    // CHECK-SAME:  }>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NCEReducesSum
func.func @NCEReducesSum(%arg0: tensor<1x64x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}>)
        -> tensor<1x1x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}> {

    %0 = VPU.NCE.Reduce(%arg0) {
                axes = [1],
                op_type = #VPU.reduce_type<SUM>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                    scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
            } -> tensor<1x1x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 28, 28] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_16x16>
    }

    return %0 : tensor<1x1x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK:       VPUIP.NCEClusterTask <{
    // CHECK-SAME:      is_superdense,
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<REDUCESUM>
    // CHECK-SAME:  }>
}

// -----

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
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
            scale = 5.000000e-01 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
    } -> tensor<1x4x224x224xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 224, 224] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_16x16>
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

    // CHECK:       PPETask {ppe = #VPU.PPEFp<
    // CHECK-SAME:      mode = <ADD>,
    // CHECK-SAME:      clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:      scale = 5.000000e-01 : f64,
    // CHECK-SAME:      prelu_alpha = [1.000000e+00],
    // CHECK-SAME:      adder = 0.000000e+00 : f64
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
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = 0.000000e+00 : f64, clamp_high = 2.550000e+02 : f64,
            scale = 5.000000e-01 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
    } -> tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 224, 224] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_16x16>
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

    // CHECK:       PPETask {ppe = #VPU.PPEFp<
    // CHECK-SAME:      mode = <ADD>,
    // CHECK-SAME:      clamp_low = 0.000000e+00 : f64, clamp_high = 2.550000e+02 : f64,
    // CHECK-SAME:      scale = 5.000000e-01 : f64,
    // CHECK-SAME:      prelu_alpha = [1.000000e+00],
    // CHECK-SAME:      adder = 0.000000e+00 : f64
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
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = 0.000000e+00 : f64, clamp_high = 2.550000e+02 : f64,
            scale = 3.0517578125E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64,
            in1_mult = [1.638400e+04], in2_mult = [1.638400e+04]>
    } -> tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 224, 224] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_16x16>
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

    // CHECK:       PPETask {ppe = #VPU.PPEFp<
    // CHECK-SAME:      mode = <ADD>,
    // CHECK-SAME:      clamp_low = 0.000000e+00 : f64, clamp_high = 2.550000e+02 : f64,
    // CHECK-SAME:      scale = 3.0517578125E-5 : f64, prelu_alpha = [1.000000e+00],
    // CHECK-SAME:      bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64,
    // CHECK-SAME:      in1_mult = [1.638400e+04], in2_mult = [1.638400e+04]
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
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = 0.000000e+00 : f64, clamp_high = 2.550000e+02 : f64,
                scale = 5.000000e-01 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>
        } -> !OutputTensor {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 256, 224] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_16x16>
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

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NceConvIDUAutopad
module @NceConvIDUAutopad {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
    }

    // CHECK:       func.func @main
    // CHECK-SAME:    [[INPUT:%.+]]: memref<1x3x16x16xf16, {order = #NHWC}, @CMX_NN>
    // CHECK-SAME:    [[WEIGHTS:%.+]]: memref<16x1x1x16xf16, {order = #NHWC}, @CMX_NN>
    // CHECK-SAME:    [[WEIGHT_TABLE:%.+]]: memref<16x1x1x4xsi32, @CMX_NN>
    // CHECK-SAME:    -> memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>
    func.func @main(%input: tensor<1x3x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                    %weights: tensor<16x1x1x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                    %weight_table: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}>)
            -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {

        %0 = VPU.NCE.Convolution(%input, %weights, %weight_table) rawFilterShape [16, 3, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,

                    strides = [1, 1]
                } : tensor<1x3x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<16x1x1x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}>
                  -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 16, 16] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
        }

        return %0 : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

        // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>

        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:      cm_sp_pattern = 7 : i64
        // CHECK-SAME:      task_type = #VPUIP.nce_task_type<CONV>
        // CHECK-SAME:  input([[INPUT]] : memref<1x3x16x16xf16, {order = #NHWC}, @CMX_NN>)
        // CHECK-SAME:  weights([[WEIGHTS]] : memref<16x1x1x16xf16, {order = #NHWC}, @CMX_NN>)
        // CHECK-SAME:  weight_table([[WEIGHT_TABLE]] : memref<16x1x1x4xsi32, @CMX_NN>)
        // CHECK-SAME:  parent_input([[INPUT]] : memref<1x3x16x16xf16, {order = #NHWC}, @CMX_NN>)
        // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
        // CHECK-SAME:  outputs([[OUT_BUF]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.19364075006223191:128>

!InterpolateSparse = !VPU.SparseTensor<
    data=tensor<1x64x64x64x!qElemType, {order = #NHWC, mem_space = @CMX_NN}>,
    storage_element_table=tensor<1x1x130x130xi32, {order = #NHWC, mem_space = @CMX_NN}>,
    #VPU.SEInterpolate<
        mode = <BILINEAR>,
        coordinate_transformation_mode = <HALF_PIXEL>,
        scale = [1.0, 1.0, 2.0, 2.0],
        offsets = [0, 0, 0, 0],
        sizes = [1, 64, 130, 130]
    >
>

// CHECK-LABEL: @SparseInt8DepthConvWithL1Opt
// CHECK-SAME: [[ARG0:%.+]]: !VPUIP.SparseBuffer<
// CHECK-SAME:      data=memref<1x64x64x64x!qElemType, {order = #NHWC}, @CMX_NN>
// CHECK-SAME:      storage_element_table=memref<1x1x130x130xi32, {order = #NHWC}, @CMX_NN>
// CHECK-SAME: [[ARG1:%.+]]: memref<64x16x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
// CHECK-SAME: [[ARG2:%.+]]: memref<64x1x1x4xsi32, @CMX_NN>
// CHECK-SAME: -> memref<1x64x128x128x!qElemType, {order = #NHWC}, @CMX_NN>
func.func @SparseInt8DepthConvWithL1Opt(
    %data: !InterpolateSparse,
    %weights: tensor<64x16x1x1x!qElemType, {order = #NHWC, mem_space = @CMX_NN}>,
    %weight_table: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>
) -> tensor<1x64x128x128x!qElemType, {order = #NHWC, mem_space = @CMX_NN}> {

    %0 = VPU.NCE.DepthConvolution(%data, %weights, %weight_table) rawFilterShape [64, 1, 3, 3] {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.280000e+02 : f64, clamp_high = 1.270000e+02 : f64,
            scale = 0.073115045214323729 : f64, prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64, adder = 1.280000e+02 : f64>,

        strides = [1, 1]
    } -> tensor<1x64x128x128x!qElemType, {order = #NHWC, mem_space = @CMX_NN}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 128, 128] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_16x16>
    }

    return %0 : tensor<1x64x128x128x!qElemType, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x64x128x128x!qElemType, {order = #NHWC}, @CMX_NN>

    // CHECK: [[DATA:%.+]], [[SE_TABLE:%.+]] = VPUIP.UngroupSparseBuffer([[ARG0]]) {resultSegmentSizes = array<i32: 1, 0, 1>}
    // CHECK-SAME: -> memref<1x64x64x64x!qElemType, {order = #NHWC}, @CMX_NN>, memref<1x1x130x130xi32, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[RES:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      is_small_kernel_optimized
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<DWCONV>
    // CHECK-SAME:  input([[DATA]] : memref<1x64x64x64x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  input_storage_element_table([[SE_TABLE]] : memref<1x1x130x130xi32, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  weights([[ARG1]] : memref<64x16x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  weight_table([[ARG2]] : memref<64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:  parent_input([[DATA]] : memref<1x64x64x64x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  parent_input_storage_element_table([[SE_TABLE]] : memref<1x1x130x130xi32, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<1x64x128x128x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUT_BUF]] : memref<1x64x128x128x!qElemType, {order = #NHWC}, @CMX_NN>)

    // CHECK: return [[RES]] : memref<1x64x128x128x!qElemType, {order = #NHWC}, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.19364075006223191:128>

!InterpolateSparse = !VPU.SparseTensor<
    data=tensor<1x64x64x64x!qElemType, {order = #NHWC, mem_space = @CMX_NN}>,
    storage_element_table=tensor<1x1x130x130xi32, {order = #NHWC, mem_space = @CMX_NN}>,
    #VPU.SEInterpolate<
        mode = <BILINEAR>,
        coordinate_transformation_mode = <HALF_PIXEL>,
        scale = [1.0, 1.0, 2.0, 2.0],
        offsets = [0, 0, 0, 0],
        sizes = [1, 64, 130, 130]
    >
>

// CHECK-LABEL: @SparseDepthConvWithoutL1Opt
// CHECK-SAME: [[ARG0:%.+]]: !VPUIP.SparseBuffer<
// CHECK-SAME:      data=memref<1x64x64x64x!qElemType, {order = #NHWC}, @CMX_NN>
// CHECK-SAME:      storage_element_table=memref<1x1x130x130xi32, {order = #NHWC}, @CMX_NN>
// CHECK-SAME: [[ARG1:%.+]]: memref<64x16x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
// CHECK-SAME: [[ARG2:%.+]]: memref<64x1x1x4xsi32, @CMX_NN>
// CHECK-SAME: -> memref<1x64x128x128x!qElemType, {order = #NHWC}, @CMX_NN>
func.func @SparseDepthConvWithoutL1Opt(
    %data: !InterpolateSparse,
    %weights: tensor<64x16x1x1x!qElemType, {order = #NHWC, mem_space = @CMX_NN}>,
    %weight_table: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>
) -> tensor<1x64x128x128x!qElemType, {order = #NHWC, mem_space = @CMX_NN}> {

    %0 = VPU.NCE.DepthConvolution(%data, %weights, %weight_table) rawFilterShape [64, 1, 3, 3] {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.280000e+02 : f64, clamp_high = 1.270000e+02 : f64,
            scale = 0.073115045214323729 : f64, prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64, adder = 1.280000e+02 : f64>,

        strides = [1, 1]
    } -> tensor<1x64x128x128x!qElemType, {order = #NHWC, mem_space = @CMX_NN}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 64, 128] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_16x16>
        VPU.DPU.Workload outOffsets [0, 0, 64, 0] outSizes [1, 64, 64, 128] pad [0, 0, 0, 0] #VPU.mpe_mode<CUBOID_16x16>
    }

    return %0 : tensor<1x64x128x128x!qElemType, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x64x128x128x!qElemType, {order = #NHWC}, @CMX_NN>

    // CHECK: [[DATA:%.+]], [[SE_TABLE:%.+]] = VPUIP.UngroupSparseBuffer([[ARG0]]) {resultSegmentSizes = array<i32: 1, 0, 1>}
    // CHECK-SAME: -> memref<1x64x64x64x!qElemType, {order = #NHWC}, @CMX_NN>, memref<1x1x130x130xi32, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[RES:%.+]] = VPUIP.NCEClusterTask
    // CHECK-NOT:       is_small_kernel_optimized
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<DWCONV>
    // CHECK-SAME:  input([[DATA]] : memref<1x64x64x64x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  input_storage_element_table([[SE_TABLE]] : memref<1x1x130x130xi32, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  weights([[ARG1]] : memref<64x16x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  weight_table([[ARG2]] : memref<64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:  parent_input([[DATA]] : memref<1x64x64x64x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  parent_input_storage_element_table([[SE_TABLE]] : memref<1x1x130x130xi32, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<1x64x128x128x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUT_BUF]] : memref<1x64x128x128x!qElemType, {order = #NHWC}, @CMX_NN>)

    // CHECK: return [[RES]] : memref<1x64x128x128x!qElemType, {order = #NHWC}, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.19364075006223191:128>

// CHECK-LABEL: @Int8ActDepthConvWithL1Opt
// CHECK-SAME: [[INPUT:%.+]]: memref<1x64x64x64x!qElemType, {order = #NHWC}, @CMX_NN>
// CHECK-SAME: [[WEIGHTS:%.+]]: memref<64x16x1x1x!qElemType, {order = #NHWC}, @CMX_NN>
// CHECK-SAME: [[WEIGHT_TABLE:%.+]]: memref<64x1x1x4xsi32, @CMX_NN>
// CHECK-SAME: -> memref<1x64x64x64x!qElemType, {order = #NHWC}, @CMX_NN>
func.func @Int8ActDepthConvWithL1Opt(
    %input: tensor<1x64x64x64x!qElemType, {order = #NHWC, mem_space = @CMX_NN}>,
    %weights: tensor<64x16x1x1x!qElemType, {order = #NHWC, mem_space = @CMX_NN}>,
    %weight_table: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>
) -> tensor<1x64x64x64x!qElemType, {order = #NHWC, mem_space = @CMX_NN}> {

    %0 = VPU.NCE.DepthConvolution(%input, %weights, %weight_table) rawFilterShape [64, 1, 3, 3] {
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -1.280000e+02 : f64, clamp_high = 1.270000e+02 : f64,
            scale = 0.073115045214323729 : f64, prelu_alpha = [1.000000e+00],
            bias = 0.000000e+00 : f64, adder = 1.280000e+02 : f64>,

        strides = [1, 1]
    } -> tensor<1x64x64x64x!qElemType, {order = #NHWC, mem_space = @CMX_NN}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 64, 64] pad [1, 1, 1, 1] #VPU.mpe_mode<CUBOID_16x16>
    }

    return %0 : tensor<1x64x64x64x!qElemType, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x64x64x64x!qElemType, {order = #NHWC}, @CMX_NN>

    // CHECK:       [[RES:%.+]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      is_small_kernel_optimized
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<DWCONV>
    // CHECK-SAME:  input([[INPUT]] : memref<1x64x64x64x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  weights([[WEIGHTS]] : memref<64x16x1x1x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  weight_table([[WEIGHT_TABLE]] : memref<64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:  parent_input([[INPUT]] : memref<1x64x64x64x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<1x64x64x64x!qElemType, {order = #NHWC}, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUT_BUF]] : memref<1x64x64x64x!qElemType, {order = #NHWC}, @CMX_NN>)

    // CHECK: return [[RES]] : memref<1x64x64x64x!qElemType, {order = #NHWC}, @CMX_NN>
}
