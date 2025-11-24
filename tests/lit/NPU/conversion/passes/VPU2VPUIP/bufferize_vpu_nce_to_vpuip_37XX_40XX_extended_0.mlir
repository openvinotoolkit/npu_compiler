//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="vpu-arch=%arch%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NceConv
// CHECK-SAME: (
// CHECK-SAME: [[ARG0:%.+]]: memref<1x16x16x16xf16, #NHWC, @CMX_NN>
// CHECK-SAME: [[ARG1:%.+]]: memref<16x16x1x1xf16, #NHWC, @CMX_NN>
// CHECK-SAME: [[ARG2:%.+]]: memref<16x1x1x4xsi32, @CMX_NN>
// CHECK-SAME: )
// CHECK-SAME: -> memref<1x16x16x16xf16, #NHWC, @CMX_NN>
func.func @NceConv(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                   %arg1: tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                   %arg2: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}>)
                   -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %0 = VPU.NCE.Convolution(%arg0, %arg1, %arg2) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                rawFilterShape = [16, 16, 1, 1],
                strides = [1, 1]
            } : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}> -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 16, 16] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
    }

    return %0 : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x16x16xf16, #NHWC, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:  input([[ARG0]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  weights([[ARG1]] : memref<16x16x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  weight_table([[ARG2]] : memref<16x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:  parent_input([[ARG0]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUT_BUF]] : memref<1x16x16x16xf16, #NHWC, @CMX_NN>)

    // CHECK: return
    // CHECK-SAME: memref<1x16x16x16xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NceMaxPool
// CHECK-SAME: (
// CHECK-SAME: [[ARG0:%.+]]: memref<1x16x1x4xf16, #NHWC, @CMX_NN>
// CHECK-SAME: [[ARG1:%.+]]: memref<16x1x1x4xsi32, @CMX_NN>
// CHECK-SAME: )
// CHECK-SAME: -> memref<1x16x1x4xf16, #NHWC, @CMX_NN>
func.func @NceMaxPool(%arg0: tensor<1x16x1x4xf16, {order = #NHWC, mem_space = @CMX_NN}>,
                      %arg1: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}>)
        -> tensor<1x16x1x4xf16, {order = #NHWC, mem_space = @CMX_NN}> {

    %0 = VPU.NCE.MaxPool(%arg0, %arg1) {
                kernel_size = [1, 1],
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                strides = [1, 1]
            } -> tensor<1x16x1x4xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 1, 4] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
    }

    return %0 : tensor<1x16x1x4xf16, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x1x4xf16, #NHWC, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<MAXPOOL>
    // CHECK-SAME:  input([[ARG0]] : memref<1x16x1x4xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  weight_table([[ARG1]] : memref<16x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:  parent_input([[ARG0]] : memref<1x16x1x4xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<1x16x1x4xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUT_BUF]] : memref<1x16x1x4xf16, #NHWC, @CMX_NN>)

    // CHECK: return
    // CHECK-SAME: memref<1x16x1x4xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NceDepthConv
// CHECK-SAME: (
// CHECK-SAME: [[ARG0:%.+]]: memref<1x16x40x80xf16, #NHWC, @CMX_NN>
// CHECK-SAME: [[ARG1:%.+]]: memref<16x1x4x8xf16, #NHWC, @CMX_NN>
// CHECK-SAME: [[ARG2:%.+]]: memref<16x1x1x4xsi32, @CMX_NN>
// CHECK-SAME: )
// CHECK-SAME: -> memref<1x16x37x73xf16, #NHWC, @CMX_NN>
func.func @NceDepthConv(%arg0: tensor<1x16x40x80xf16, {order = #NHWC, mem_space = @CMX_NN}>,
                        %arg1: tensor<16x1x4x8xf16, {order = #NHWC, mem_space = @CMX_NN}>,
                        %arg2: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN}>)
        -> tensor<1x16x37x73xf16, {order = #NHWC, mem_space = @CMX_NN}> {

    %0 = VPU.NCE.DepthConvolution(%arg0, %arg1, %arg2) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 0 , right = 0, top = 0, bottom = 0>,
                rawFilterShape = [16, 1, 4, 8],
                strides = [1, 1]
            } -> tensor<1x16x37x73xf16, {order = #NHWC, mem_space = @CMX_NN}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 37, 73] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
    }

    return %0 : tensor<1x16x37x73xf16, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x37x73xf16, #NHWC, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<DWCONV>
    // CHECK-SAME:  input([[ARG0]] : memref<1x16x40x80xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  weights([[ARG1]] : memref<16x1x4x8xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  weight_table([[ARG2]] : memref<16x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:  parent_input([[ARG0]] : memref<1x16x40x80xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<1x16x37x73xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUT_BUF]] : memref<1x16x37x73xf16, #NHWC, @CMX_NN>)

    // CHECK: return
    // CHECK-SAME: memref<1x16x37x73xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InterpolateTensor = !VPU.SparseTensor<
    data=tensor<1x64x5x10xf16, {order = #NHWC, mem_space = @CMX_NN}>,
    sparsity_map=tensor<1x64x11x21xi1, {mem_space = @CMX_NN}>,
    storage_element_table=tensor<1x1x11x21xi32, {order = #NHWC, mem_space = @CMX_NN}>,
    #VPU.SEInterpolate<
        mode = <BILINEAR>,
        coordinate_transformation_mode = <ASYMMETRIC>,
        scale = [1.0, 1.0, 2.0, 2.0],
        offsets = [0, 0, 0, 0],
        sizes = [1, 64, 11, 21]
    >
>

// CHECK-LABEL: @NceInterpolate
// CHECK-SAME: (
// CHECK-SAME: [[ARG0:%.+]]: !VPUIP.SparseBuffer<
// CHECK-SAME:      data=memref<1x64x5x10xf16, #NHWC, @CMX_NN>
// CHECK-SAME:      sparsity_map=memref<1x64x11x21xi1, @CMX_NN>
// CHECK-SAME:      storage_element_table=memref<1x1x11x21xi32, #NHWC, @CMX_NN>
// CHECK-SAME: [[ARG1:%.+]]: memref<64x64x1x1xf16, #NHWC, @CMX_NN>
// CHECK-SAME: [[ARG2:%.+]]: memref<64x1x1x4xsi32, @CMX_NN>
// CHECK-SAME: )
// CHECK-SAME: -> memref<1x64x10x20xf16, #NHWC, @CMX_NN>
func.func @NceInterpolate(
    %data: !InterpolateTensor,
    %weights: tensor<64x64x1x1xf16, {order = #NHWC, mem_space = @CMX_NN}>,
    %weight_table: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>
) -> tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}> {

    %0 = VPU.NCE.Interpolate(%data, %weights, %weight_table) {
        rawFilterShape = [64, 64, 2, 2],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 10, 20] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
    }

    return %0 : tensor<1x64x10x20xf16, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x64x10x20xf16, #NHWC, @CMX_NN>

    // CHECK: [[DATA:%.+]], [[SPARSITY_MAP:%.+]], [[STORAGE_ELEMENT:%.+]] = VPUIP.UngroupSparseBuffer([[ARG0]]) {
    // CHECK-SAME: ->
    // CHECK-SAME:  memref<1x64x5x10xf16, #NHWC, @CMX_NN>,
    // CHECK-SAME:  memref<1x64x11x21xi1, @CMX_NN>,
    // CHECK-SAME:  memref<1x1x11x21xi32, #NHWC, @CMX_NN>

    // CHECK:      VPUIP.NCEClusterTask
    // CHECK-SAME:     task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME: input([[DATA]] : memref<1x64x5x10xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: input_sparsity_map([[SPARSITY_MAP]] : memref<1x64x11x21xi1, @CMX_NN>)
    // CHECK-SAME: input_storage_element_table([[STORAGE_ELEMENT]] : memref<1x1x11x21xi32, #NHWC, @CMX_NN>)
    // CHECK-SAME: weights([[ARG1]] : memref<64x64x1x1xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: weight_table([[ARG2]] : memref<64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME: parent_input([[DATA]] : memref<1x64x5x10xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: parent_input_sparsity_map([[SPARSITY_MAP]] : memref<1x64x11x21xi1, @CMX_NN>)
    // CHECK-SAME: parent_input_storage_element_table([[STORAGE_ELEMENT]] : memref<1x1x11x21xi32, #NHWC, @CMX_NN>)
    // CHECK-SAME: parent_output([[OUT_BUF]] : memref<1x64x10x20xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME: outputs([[OUT_BUF]] : memref<1x64x10x20xf16, #NHWC, @CMX_NN>)

    // CHECK: return
    // CHECK-SAME: memref<1x64x10x20xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NceEltwiseAdd
// CHECK-SAME: (
// CHECK-SAME: [[ARG0:%.+]]: memref<1x64x28x28xf16, #NHWC, @CMX_NN>,
// CHECK-SAME: [[ARG1:%.+]]: memref<1x64x28x28xf16, #NHWC, @CMX_NN>
// CHECK-SAME: )
// CHECK-SAME: -> memref<1x64x28x28xf16, #NHWC, @CMX_NN>
func.func @NceEltwiseAdd(%arg0: tensor<1x64x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}>,
                         %arg1: tensor<1x64x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}> {

    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
                op_type = #VPU.eltwise_type<ADD>,
                ppe = #VPU.PPEStub<>
            } -> tensor<1x64x28x28xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 28, 28] <left = 0 , right = 0, top = 0, bottom = 0> #VPU.mpe_mode<VECTOR_FP16>
    }

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x64x28x28xf16, #NHWC, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask
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

// CHECK-LABEL: @NceCompressConv
// CHECK-SAME: (
// CHECK-SAME: [[ARG0:%.+]]: memref<1x4x112x112xf16, #NHWC, @CMX_NN>
// CHECK-SAME: [[ARG1:%.+]]: memref<64x4x7x7xf16, #NHWC, @CMX_NN>
// CHECK-SAME: [[ARG2:%.+]]: memref<64x1x1x4xsi32, @CMX_NN>
// CHECK-SAME: )
// CHECK-SAME: -> memref<1x64x56x56xf16, #NHWC, @CMX_NN>
func.func @NceCompressConv(%arg0: tensor<1x4x112x112xf16, {order = #NHWC, mem_space = @CMX_NN}>,
                           %arg1: tensor<64x4x7x7xf16, {order = #NHWC, mem_space = @CMX_NN}>,
                           %arg2: tensor<64x1x1x4xsi32, {mem_space = @CMX_NN}>)
        -> tensor<1x64x56x56xf16, {order = #NHWC, mem_space = @CMX_NN}> {

    %0 = VPU.NCE.CompressConvolution(%arg0, %arg1, %arg2) {
            cm_sp_pattern = 15 : i64,
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
            rawFilterShape = [64, 4, 7, 7], strides = [2, 2]
        } -> tensor<1x64x56x56xf16, {order = #NHWC, mem_space = @CMX_NN}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 56, 56] <left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 0 : i64> #VPU.mpe_mode<CUBOID_16x16> attributes {cluster_id = 0 : i64}
    }

    return %0 : tensor<1x64x56x56xf16, {order = #NHWC, mem_space = @CMX_NN}>

    // Note: VPU.NCE.CompressConvolution is "special" as there are shape casts:

    // CHECK: [[SHAPE_CAST_ARG1:%.+]] = VPUIP.ShapeCast {shape = [64, 16, 7, 7]} inputs([[ARG1]] : memref<64x4x7x7xf16, #NHWC, @CMX_NN>) -> memref<64x16x7x7xf16, #NHWC, @CMX_NN>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x64x56x56xf16, #NHWC, @CMX_NN>

    // CHECK: [[SHAPE_CAST_ARG0:%.+]] = VPUIP.ShapeCast {shape = [1, 16, 112, 112]} inputs([[ARG0]] : memref<1x4x112x112xf16, #NHWC, @CMX_NN>) -> memref<1x16x112x112xf16, #NHWC, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:  input([[SHAPE_CAST_ARG0]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  weights([[SHAPE_CAST_ARG1]] : memref<64x16x7x7xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  weight_table([[ARG2]] : memref<64x1x1x4xsi32, @CMX_NN>)
    // CHECK-SAME:  parent_input([[SHAPE_CAST_ARG0]] : memref<1x16x112x112xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  outputs([[OUT_BUF]] : memref<1x64x56x56xf16, #NHWC, @CMX_NN>)

    // CHECK: return
    // CHECK-SAME: memref<1x64x56x56xf16, #NHWC, @CMX_NN>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NceAvgPool
// CHECK-SAME: (
// CHECK-SAME: [[ARG0:%.+]]: memref<1x16x15x15xf16, #NHWC, @CMX_NN>
// CHECK-SAME: )
// CHECK-SAME: -> memref<1x16x15x15xf16, #NHWC, @CMX_NN>
func.func @NceAvgPool(%arg0: tensor<1x16x15x15xf16, {order = #NHWC, mem_space = @CMX_NN}>)
        -> tensor<1x16x15x15xf16, {order = #NHWC, mem_space = @CMX_NN}> {

    %0 = VPU.NCE.AveragePool(%arg0) {
        kernel_size = [1, 1],
        minimumHardwareExecutionCost = 708 : i64,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } -> tensor<1x16x15x15xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 15, 15] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 13 : i64> #VPU.mpe_mode<CUBOID_16x16>
    }

    return %0 : tensor<1x16x15x15xf16, {order = #NHWC, mem_space = @CMX_NN}>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<1x16x15x15xf16, #NHWC, @CMX_NN>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<AVEPOOL>
    // CHECK-SAME:  input([[ARG0]] : memref<1x16x15x15xf16, #NHWC, @CMX_NN>)
    // CHECK-SAME:  output([[OUT_BUF]] : memref<1x16x15x15xf16, #NHWC, @CMX_NN>)

    // CHECK: return
    // CHECK-SAME: memref<1x16x15x15xf16, #NHWC, @CMX_NN>
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
            ppe = #VPU.PPEStub<>
        } -> !OutputTensor {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 3, 256, 224] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_16x16>
        }

    return %out : !OutputTensor

    // CHECK:       [[VIEW_OP_IN:%.*]] = VPUIP.ViewOp [[ARG0]] :
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x3x256x224xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:    to
    // CHECK-SAME:    !VPUIP.DistributedBuffer<1x224x3x256xf16, #NHWC, @CMX_NN,

    // CHECK:       [[OUT_BUF:%.*]] = VPURT.AllocDistributed
    // CHECK-SAME:    -> !VPUIP.DistributedBuffer<1x224x4x256x!qElemType, #NWCH, @CMX_NN,
    // CHECK-SAME:   {mode = "OVERLAPPED", num_tiles = [1, 1, 1, 2], kernel = [1, 1],
    // CHECK-SAME:   pads = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 1 : i64>,
    // CHECK-SAME:   strides = [1, 2], num_clusters = 2 : i64, equal_memory_and_compute_view}>

    // CHECK:        [[MULTI_TILE_RES:%.*]] = VPUIP.NCEClusterTask
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

// CHECK-LABEL: @SparseInPlaceNCEEltwise
// CHECK-SAME:  [[ARG0:%.+]]: !VPUIP.SparseBuffer<data=memref<1x32x368x29xf16, #NHWC, @CMX_NN>, sparsity_map=memref<1x64x368x29xi1, #NHWC, @CMX_NN>>
func.func @SparseInPlaceNCEEltwise(%arg0: !VPU.SparseTensor<data=tensor<1x32x368x29xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                                                                sparsity_map=tensor<1x64x368x29xi1, {mem_space = @CMX_NN, order = #NHWC}>>)
        -> tensor<1x32x368x29xf16, {mem_space = @CMX_NN, order = #NHWC}> {

    %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
                op_type = #VPU.eltwise_type<ADD>,
                ppe = #VPU.PPEStub<>
            } -> tensor<1x32x368x29xf16, {mem_space = @CMX_NN, order = #NHWC}> {
                VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 64, 123, 29] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64> <CUBOID_8x16> attributes {cluster_id = 0 : i64}
            }

    return %0 : tensor<1x32x368x29xf16, {mem_space = @CMX_NN, order = #NHWC}>

    //CHECK:      [[ALLOC:%.+]] = memref.alloc() : memref<1x32x368x29xf16, #NHWC, @CMX_NN>
    //CHECK:      [[DATA2:%.+]], [[SM2:%.+]] = VPUIP.UngroupSparseBuffer([[ARG0]]) {resultSegmentSizes = array<i32: 1, 1, 0>} -> memref<1x32x368x29xf16, #NHWC, @CMX_NN>, memref<1x64x368x29xi1, #NHWC, @CMX_NN>
    //CHECK:      [[DATA1:%.+]], [[SM1:%.+]] = VPUIP.UngroupSparseBuffer([[ARG0]]) {resultSegmentSizes = array<i32: 1, 1, 0>} -> memref<1x32x368x29xf16, #NHWC, @CMX_NN>, memref<1x64x368x29xi1, #NHWC, @CMX_NN>

    //CHECK:        [[OUT:%.+]] = VPUIP.NCEClusterTask {
    //CHECK-SAME:       eltwise_type = #VPU.eltwise_type<ADD>,
    //CHECK-SAME:       task_type = #VPUIP.nce_task_type<ELTWISE>
    //CHECK-SAME:   }
    //CHECK-SAME:       input([[DATA2]] : memref<1x32x368x29xf16, #NHWC, @CMX_NN>)
    //CHECK-SAME:       input_sparsity_map([[SM2]]
    //CHECK-SAME:       weights([[DATA1]]
    //CHECK-SAME:       weights_sparsity_map([[SM1]]
    //CHECK-SAME:       outputs([[ALLOC]]

    //CHECK:     return [[OUT]] : memref<1x32x368x29xf16, #NHWC, @CMX_NN>
}
