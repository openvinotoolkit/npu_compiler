//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --one-shot-bufferize-VPU-to-VPUIP --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU50XX

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
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 7, 4] <left = 1 : i64, right = 0 : i64, top = 0 : i64, bottom = 13 : i64> #VPU.mpe_mode<CUBOID_16x16>
    }

    return %1 : tensor<1x16x15x13xf16, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK:       VPUIP.NCEClusterTask {
    // CHECK-NOT:       is_small_kernel_optimized,
    // CHECK-SAME:      is_superdense,
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<AVEPOOL>
    // CHECK-SAME:  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseSubtract
// CHECK-SAME: ([[ARG0:%.+]]: memref<1x64x28x28xf16, #NHWC, @CMX_NN>,
// CHECK-SAME: [[ARG1:%.+]]: memref<1x64x28x28xf16, #NHWC, @CMX_NN>)
func.func @EltwiseSubtract(%arg0: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %1 = VPU.NCE.Eltwise(%arg0, %arg1) {
                op_type = #VPU.eltwise_type<SUBTRACT>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
            } -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 5, 28] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        VPU.DPU.Workload outOffsets [0, 0, 5, 0] outSizes [1, 64, 5, 28] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        VPU.DPU.Workload outOffsets [0, 0, 10, 0] outSizes [1, 64, 5, 28] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        VPU.DPU.Workload outOffsets [0, 0, 15, 0] outSizes [1, 64, 5, 28] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        VPU.DPU.Workload outOffsets [0, 0, 20, 0] outSizes [1, 64, 8, 28] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
    }
    return %1 : tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[ALLOC0:%.+]] = memref.alloc() : memref<1x64x28x28xf16, #NHWC, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK:               eltwise_type = #VPU.eltwise_type<SUBTRACT>
    // CHECK-SAME:          task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:      input([[ARG0]] : memref<1x64x28x28xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      weights([[ARG1]] : memref<1x64x28x28xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_input([[ARG0]] : memref<1x64x28x28xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      parent_output([[ALLOC0]] : memref<1x64x28x28xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:      outputs([[ALLOC0]] : memref<1x64x28x28xf16, #NHWC, @CMX_NN>)
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
// CHECK-SAME: [[ARG0:%.+]]: memref<1x64x28x28xf16, #NHWC, @CMX_NN>,
// CHECK-SAME: [[ARG1:%.+]]: memref<1x64x28x28xf16, #NHWC, @CMX_NN>
// CHECK-SAME: )
// CHECK-SAME: -> memref<1x64x28x28xf16, #NHWC, @CMX_NN>
func.func @NceEltwiseMultiply(%arg0: tensor<1x64x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}>,
                         %arg1: tensor<1x64x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}> {

    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
                op_type = #VPU.eltwise_type<MULTIPLY>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
            } -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 28, 28] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
    }

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x64x28x28xf16, #NHWC, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK:               eltwise_type = #VPU.eltwise_type<MULTIPLY>
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:  input([[ARG0]] : memref<1x64x28x28xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  weights([[ARG1]] : memref<1x64x28x28xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  parent_input([[ARG0]] : memref<1x64x28x28xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<1x64x28x28xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUT_BUF]] : memref<1x64x28x28xf16, #NHWC, @CMX_NN>)

    // CHECK: return
    // CHECK-SAME: memref<1x64x28x28xf16, #NHWC, @CMX_NN>
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
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 28, 28] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16>
    }

    return %0 : tensor<1x1x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK:       VPUIP.NCEClusterTask {
    // CHECK-SAME:      is_superdense,
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<REDUCEMEAN>
    // CHECK-SAME:  }
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
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 28, 28] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<CUBOID_16x16>
    }

    return %0 : tensor<1x1x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK:       VPUIP.NCEClusterTask {
    // CHECK-SAME:      is_superdense,
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<REDUCESUM>
    // CHECK-SAME:  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @NcePermute
// CHECK-SAME: (
// CHECK-SAME: [[ARG0:%.+]]: memref<1x3x224x224xf16, @CMX_NN>
// CHECK-SAME: )
// CHECK-SAME: -> memref<1x4x224x224x!qElemType, #NHWC, @CMX_NN>
func.func @NcePermute(%arg0: tensor<1x3x224x224xf16, {mem_space = @CMX_NN}>)
        -> tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> {

    %0 = VPU.NCE.Permute(%arg0) {
        dstElemType = !qElemType,
        dstOrder = #NHWC,
        expandedChannels = 4 : i64,
        ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
            scale = 5.000000e-01 : f64, prelu_alpha = [7.000000e+00], adder = 0.000000e+00 : f64>
    } -> tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 224, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
    }

    return %0 : tensor<1x4x224x224x!qElemType, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK: [[VIEW_OP_IN:%.*]] = VPUIP.ViewOp [[ARG0]] : memref<1x3x224x224xf16, @CMX_NN>
    // CHECK-SAME:  to memref<1x224x3x224xf16, #NHWC, @CMX_NN>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x224x4x224x!qElemType, #NWCH, @CMX_NN>

    // CHECK:       [[PERMUTE_RES:%.*]] = VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>
    // CHECK-SAME:  input([[VIEW_OP_IN]] : memref<1x224x3x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  weights([[VIEW_OP_IN]] : memref<1x224x3x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  parent_input([[VIEW_OP_IN]] : memref<1x224x3x224xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<1x224x4x224x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUT_BUF]] : memref<1x224x4x224x!qElemType, #NWCH, @CMX_NN>)
    // CHECK-SAME:  -> memref<1x224x4x224x!qElemType, #NWCH, @CMX_NN>

    // CHECK:       PPETask {ppe = #VPU.PPEFp<
    // CHECK-SAME:      mode = <ADD>,
    // CHECK-SAME:      clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:      scale = 3.500000e+00 : f64,
    // CHECK-SAME:      prelu_alpha = [1.000000e+00],
    // CHECK-SAME:      adder = 0.000000e+00 : f64
    // CHECK-SAME:  >}

    // CHECK: [[VIEW_OP_OUT:%.*]] = VPUIP.ViewOp [[PERMUTE_RES]] : memref<1x224x4x224x!qElemType, #NWCH, @CMX_NN>
    // CHECK-SAME: to memref<1x4x224x224x!qElemType, #NHWC, @CMX_NN>

    // CHECK: return [[VIEW_OP_OUT]] : memref<1x4x224x224x!qElemType, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NceConvIDUAutopad
module @NceConvIDUAutopad {
    config.PipelineOptions @Options {
        config.Option @config.AutoPaddingIDU : true
    }

    // CHECK:       func.func @main
    // CHECK-SAME:    [[INPUT:%.+]]: memref<1x3x16x16xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:    [[WEIGHTS:%.+]]: memref<16x1x1x16xf16, #NHWC, @CMX_NN>
    // CHECK-SAME:    [[WEIGHT_TABLE:%.+]]: memref<16x1x1x4xsi32, @CMX_NN>
    // CHECK-SAME:    -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>
    func.func @main(%input: tensor<1x3x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                    %weights: tensor<16x1x1x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                    %weight_table: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}>)
            -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {

        %0 = VPU.NCE.Convolution(%input, %weights, %weight_table) {
                    ppe = #VPU.PPEStub<>,
                    pad = #VPU.Padding<left = 0, right = 0, top = 0, bottom = 0>,
                    rawFilterShape = [16, 3, 1, 1],
                    strides = [1, 1]
                } : tensor<1x3x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<16x1x1x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}>
                  -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 16, 16] <left = 0, right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
        }

        return %0 : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

        // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>

        // CHECK:       VPUIP.NCEClusterTask
        // CHECK-SAME:      cm_sp_pattern = 7 : i64
        // CHECK-SAME:      task_type = #VPUIP.nce_task_type<CONV>
        // CHECK-SAME:  input([[INPUT]] : memref<1x3x16x16xf16, #NHWC, @CMX_NN>)
        // CHECK-SAME:  weights([[WEIGHTS]] : memref<16x1x1x16xf16, #NHWC, @CMX_NN>)
        // CHECK-SAME:  weight_table([[WEIGHT_TABLE]] : memref<16x1x1x4xsi32, @CMX_NN>)
        // CHECK-SAME:  parent_input([[INPUT]] : memref<1x3x16x16xf16, #NHWC, @CMX_NN>)
        // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
        // CHECK-SAME:  outputs([[OUT_BUF]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    }
}
