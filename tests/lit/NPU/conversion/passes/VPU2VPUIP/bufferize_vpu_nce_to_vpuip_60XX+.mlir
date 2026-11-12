//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --verify-diagnostics --init-compiler="platform=%platform%" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: platform-NPU6010

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

// CHECK-LABEL: @MatMul
// CHECK-SAME: (
// CHECK-SAME: [[INPUT:%.+]]: memref<43x1x32x16x4xf16, {order = #GNHWC}, [@CMX_NN, 0]>
// CHECK-SAME: [[WEIGHTS:%.+]]: memref<43x64x32x1x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>
// CHECK-SAME: [[SCALE_TABLE:%.+]]: memref<43x64x1x1x1xf32, [@CMX_NN, 0]>, [[BIAS_TABLE:%.+]]: memref<43x64x1x1x1xf32, [@CMX_NN, 0]>
// CHECK-SAME: )
// CHECK-SAME: -> memref<43x1x64x16x4xf16, {order = #GNHWC}, [@CMX_NN, 0]>
func.func @MatMul(%arg0: tensor<43x1x32x16x4xf16, {mem_space = [@CMX_NN, 0], order = #GNHWC}>,
                   %arg1: tensor<43x64x32x1x1xf16, {mem_space = [@CMX_NN, 0], order = #GNHWC}>,
                   %arg2: tensor<43x64x1x1x1xf32, {mem_space = [@CMX_NN, 0], order = #NCDHW}>,
                   %arg3: tensor<43x64x1x1x1xf32, {mem_space = [@CMX_NN, 0], order = #NCDHW}>)
                   -> tensor<43x1x64x16x4xf16, {mem_space = [@CMX_NN, 0], order = #GNHWC}> {

    %0 = VPU.NCE.MatMul(%arg0, %arg1, %arg2, %arg3) rawFilterShape [43, 64, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                mpe_engine = #VPU.MPEEngine60XX<mode = <SCL>>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,

                strides = [1, 1],
                operandSegmentSizes = array<i32: 1, 1, 0, 1, 1, 0, 0>
            } -> tensor<43x1x64x16x4xf16, {mem_space = [@CMX_NN, 0], order = #GNHWC}> {
        VPU.DPU.Workload inOffsets [0, 0, 0, 0, 0] inSizes [43, 1, 32, 16, 4] outOffsets [0, 0, 0, 0, 0] outSizes [43, 1, 64, 16, 4] pad [0, 0, 0, 0, 0] <CUBOID_16x16> attributes {cluster_id = 0 : i64}
    }

    return %0 : tensor<43x1x64x16x4xf16, {mem_space = [@CMX_NN, 0], order = #GNHWC}>

    // CHECK: [[OUT_BUF:%.+]] = memref.alloc() : memref<43x1x64x16x4xf16, {order = #GNHWC}, [@CMX_NN, 0]>

    // CHECK:       VPUIP.NCEClusterTask
    // CHECK-SAME:      task_type = #VPUIP.nce_task_type<CONV>
    // CHECK-SAME:  input([[INPUT]] : memref<43x1x32x16x4xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:  weights([[WEIGHTS]] : memref<43x64x32x1x1xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:  weight_table_scale([[SCALE_TABLE]] : memref<43x64x1x1x1xf32, [@CMX_NN, 0]>)
    // CHECK-SAME:  weight_table_bias([[BIAS_TABLE]] : memref<43x64x1x1x1xf32, [@CMX_NN, 0]>)
    // CHECK-SAME:  parent_input([[INPUT]] : memref<43x1x32x16x4xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:  parent_output([[OUT_BUF]] : memref<43x1x64x16x4xf16, {order = #GNHWC}, [@CMX_NN, 0]>)
    // CHECK-SAME:  outputs([[OUT_BUF]] : memref<43x1x64x16x4xf16, {order = #GNHWC}, [@CMX_NN, 0]>)

    // CHECK: return
    // CHECK-SAME: memref<43x1x64x16x4xf16, {order = #GNHWC}, [@CMX_NN, 0]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @ConvReduceMax
// CHECK-SAME:      [[INPUT0:%.+]]:  memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:      [[INPUT1:%.+]]: memref<16x16x1x1xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:                      -> (memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:                       memref<1x16x16x1xf16, {order = #NHWC}, @CMX_NN>) {
func.func @ConvReduceMax(%arg0: tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                   %arg1: tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
                   -> (tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                       tensor<1x16x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>) {

    %conv, %reduce_xy_max = VPU.NCE.Convolution(%arg0, %arg1) rawFilterShape [16, 16, 1, 1] {
                                    axes_value = [3],
                                    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                                    ppe = #VPU.PPEStub<>,
                                    reduce_extra_outputs = 1 : i32,
                                    strides = [1, 1],
                                    resultSegmentSizes = array<i32: 1, 1, 0, 0>}
                             : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>, tensor<16x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
                              -> tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                 tensor<1x16x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 16, 16] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
    }
    return %conv, %reduce_xy_max : tensor<1x16x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                   tensor<1x16x16x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
// CHECK:      [[OUT0:%.+]] = memref.alloc() : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>
// CHECK:      [[OUT1:%.+]] = memref.alloc() : memref<1x16x16x1xf16, {order = #NHWC}, @CMX_NN>
// CHECK:      [[CTASK_RES:%.+]]:2 = VPUIP.NCEClusterTask
// CHECK-SAME:                resultSegmentSizes = array<i32: 1, 0, 0, 1, 0, 0>, task_type = #VPUIP.nce_task_type<CONV>}
// CHECK-SAME:                input([[INPUT0]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:                weights([[INPUT1]] : memref<16x16x1x1xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:                parent_input([[INPUT0]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:                parent_output([[OUT0]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:                outputs([[OUT0]] : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:                max_per_xy_buff([[OUT1]] : memref<1x16x16x1xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:                -> memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:                   memref<1x16x16x1xf16, {order = #NHWC}, @CMX_NN> variants : {

// CHECK:    return [[CTASK_RES]]#0, [[CTASK_RES]]#1 : memref<1x16x16x16xf16, {order = #NHWC}, @CMX_NN>, memref<1x16x16x1xf16, {order = #NHWC}, @CMX_NN>

}

// -----


#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!INPUT_CMX = tensor<1x32x16x16xf16,
    {mem_space = @CMX_NN, order = #NHWC}>

!WEIGHTS_CMX = tensor<64x32x3x3xf16,
    {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       func.func @SparseNCEConvReduceMax
// CHECK-SAME:              ([[ARG0:%.+]]: memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:               [[ARG1:%.+]]: memref<64x32x3x3xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:       -> (memref<1x64x14x14xf16, {order = #NHWC}>
// CHECK-SAME:          memref<1x64x14x1xf16, {order = #NHWC}>) {

func.func @SparseNCEConvReduceMax(%input_sparse_cmx : !INPUT_CMX,
                                  %weights_sparse_cmx : !WEIGHTS_CMX)
        -> (tensor<1x64x14x14xf16, {order = #NHWC}>,
            tensor<1x64x14x1xf16, {order = #NHWC}>) {

    %output_sparse_cmx, %reduce_xy_max = VPU.NCE.Convolution(%input_sparse_cmx, %weights_sparse_cmx) rawFilterShape [64, 32, 3, 3] {
            axes_value = [3],
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            strides = [1, 1],
            reduce_extra_outputs = 1 : i32,
            resultSegmentSizes = array<i32: 1, 1, 0, 0>}
        : tensor<1x32x16x16xf16, {mem_space = @CMX_NN, order = #NHWC}>,
          tensor<64x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<1x64x14x1xf16, {mem_space = @CMX_NN, order = #NHWC}> {
            VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 16, 16] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
        }
    %output_sparse = VPU.Copy(%output_sparse_cmx) :
           tensor<1x64x14x14xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x64x14x14xf16, {order = #NHWC}>

    %output_reduce_xy_max = VPU.Copy(%reduce_xy_max) : tensor<1x64x14x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
        -> tensor<1x64x14x1xf16, {order = #NHWC}>

    return %output_sparse, %output_reduce_xy_max : tensor<1x64x14x14xf16, {order = #NHWC}>,
                                                   tensor<1x64x14x1xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT:%.+]] = memref.alloc() : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>
// CHECK:       [[MAX_PER_XY:%.+]] = memref.alloc() : memref<1x64x14x1xf16, {order = #NHWC}, @CMX_NN>
// CHECK:       [[NCE_CT:%.+]]:2 = VPUIP.NCEClusterTask
// CHECK-SAME:            resultSegmentSizes = array<i32: 1, 0, 0, 1, 0, 0>, task_type = #VPUIP.nce_task_type<CONV>}>
// CHECK-SAME:            input([[ARG0]] : memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:            weights([[ARG1]] : memref<64x32x3x3xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:            parent_input([[ARG0]] : memref<1x32x16x16xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:            parent_output([[OUTPUT]] : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:            outputs([[OUTPUT]] : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:            max_per_xy_buff([[MAX_PER_XY]] : memref<1x64x14x1xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:            -> memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:               memref<1x64x14x1xf16, {order = #NHWC}, @CMX_NN>
// CHECK:       [[DATA_DDR:%.+]] = memref.alloc() : memref<1x64x14x14xf16, {order = #NHWC}>
// CHECK:       [[NCE_CT_OUT:%.+]] = VPUIP.Copy inputs([[NCE_CT]]#0 : memref<1x64x14x14xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:               outputs([[DATA_DDR]] : memref<1x64x14x14xf16, {order = #NHWC}>) -> memref<1x64x14x14xf16, {order = #NHWC}>
// CHECK:       [[MAX_PER_XY_DDR:%.+]] = memref.alloc() : memref<1x64x14x1xf16, {order = #NHWC}>
// CHECK:       [[MAX_PER_XY_OUT:%.+]] = VPUIP.Copy inputs([[NCE_CT]]#1 : memref<1x64x14x1xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:               outputs([[MAX_PER_XY_DDR]] : memref<1x64x14x1xf16, {order = #NHWC}>) -> memref<1x64x14x1xf16, {order = #NHWC}>
// CHECK:       return [[NCE_CT_OUT]], [[MAX_PER_XY_OUT]]
// CHECK-SAME:              : memref<1x64x14x14xf16, {order = #NHWC}>,
// CHECK-SAME:                memref<1x64x14x1xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @EltwiseReduceMax
// CHECK-SAME:      [[INPUT0:%.+]]: memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:      [[INPUT1:%.+]]: memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:      -> (memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:          memref<1x1x8x8xf16, {order = #NHWC}, @CMX_NN>) {
func.func @EltwiseReduceMax(%arg0: tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                            %arg1: tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    -> (tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
    %eltwise, %reduce_xy_max = VPU.NCE.Eltwise(%arg0, %arg1) {
            axes_value = [1],
            op_type = #VPU.eltwise_type<ADD>,
            ppe = #VPU.PPEStub<>,
            resultSegmentSizes = array<i32: 1, 1, 0, 0>
        } : tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>
          -> tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
             tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 8, 8] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
    }
    return %eltwise, %reduce_xy_max : tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                      tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[OUT0:%.+]] = memref.alloc() : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>
// CHECK:       [[OUT1:%.+]] = memref.alloc() : memref<1x1x8x8xf16, {order = #NHWC}, @CMX_NN>
// CHECK:       [[CTASK_RES:%.+]]:2 = VPUIP.NCEClusterTask
// CHECK-SAME:      task_type = #VPUIP.nce_task_type<ELTWISE>
// CHECK-SAME:      input([[INPUT0]] : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:      weights([[INPUT1]] : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:      parent_output([[OUT0]] : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:      outputs([[OUT0]] : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:      max_per_xy_buff([[OUT1]] : memref<1x1x8x8xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:      -> memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:         memref<1x1x8x8xf16, {order = #NHWC}, @CMX_NN>
// CHECK:       return [[CTASK_RES]]#0, [[CTASK_RES]]#1
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @DepthConvolutionReduceMax
// CHECK-SAME:      [[INPUT:%.+]]: memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:      [[FILTER:%.+]]: memref<32x16x1x1xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:      -> (memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:          memref<1x1x8x8xf16, {order = #NHWC}, @CMX_NN>) {
func.func @DepthConvolutionReduceMax(%arg0: tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                     %arg1: tensor<32x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>)
    -> (tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
        tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>) {
    %dwconv, %reduce_xy_max = VPU.NCE.DepthConvolution(%arg0, %arg1) rawFilterShape [32, 1, 1, 1] {
            axes_value = [1],
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            resultSegmentSizes = array<i32: 1, 1, 0, 0>,
            strides = [1, 1]
        } -> tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
             tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 32, 8, 8] pad [0, 0, 0, 0] #VPU.mpe_mode<VECTOR_FP16>
    }
    return %dwconv, %reduce_xy_max : tensor<1x32x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>,
                                     tensor<1x1x8x8xf16, {mem_space = @CMX_NN, order = #NHWC}>

// CHECK:       [[OUT0:%.+]] = memref.alloc() : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>
// CHECK:       [[OUT1:%.+]] = memref.alloc() : memref<1x1x8x8xf16, {order = #NHWC}, @CMX_NN>
// CHECK:       [[CTASK_RES:%.+]]:2 = VPUIP.NCEClusterTask
// CHECK-SAME:      task_type = #VPUIP.nce_task_type<DWCONV>
// CHECK-SAME:      input([[INPUT]] : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:      weights([[FILTER]] : memref<32x16x1x1xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:      parent_output([[OUT0]] : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:      outputs([[OUT0]] : memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:      max_per_xy_buff([[OUT1]] : memref<1x1x8x8xf16, {order = #NHWC}, @CMX_NN>)
// CHECK-SAME:      -> memref<1x32x8x8xf16, {order = #NHWC}, @CMX_NN>,
// CHECK-SAME:         memref<1x1x8x8xf16, {order = #NHWC}, @CMX_NN>
// CHECK:       return [[CTASK_RES]]#0, [[CTASK_RES]]#1
}
