//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --adjust-memory-space %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @ConvNCEtoCMX
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x16x16xf16, {order = #NHWC}>)
func.func @ConvNCEtoCMX(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %weights) rawFilterShape [16, 16, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        
        strides = [1, 1]
    } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS_DDR:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[IN_CMX:%.+]] = VPU.Copy([[ARG_0]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:       [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS_DDR]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<16x16x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.Convolution([[IN_CMX]], [[WEIGHTS_CMX]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[OUT_DDR:%.+]] = VPU.Copy([[OUT_CMX]])
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[OUT_DDR]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @DepthConvNCEtoCMX
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x40x80xf16, {order = #NHWC}>)
func.func @DepthConvNCEtoCMX(%arg0: tensor<1x16x40x80xf16, {order = #NHWC}>) -> tensor<1x16x37x73xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x1x4x8xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.DepthConvolution(%arg0, %weights) rawFilterShape [16, 1, 4, 8] {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        
        strides = [1, 1]
    } -> tensor<1x16x37x73xf16, {order = #NHWC}>

    return %0 : tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_DDR:%.+]] = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}>

    // CHECK:       [[IN_CMX:%.+]] = VPU.Copy([[ARG_0]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<1x16x40x80xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:       [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS_DDR]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<16x1x4x8xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.DepthConvolution([[IN_CMX]], [[WEIGHTS_CMX]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[OUT_DDR:%.+]] = VPU.Copy([[OUT_CMX]])
    // CHECK-SAME:      -> tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK:       return [[OUT_DDR]] : tensor<1x16x37x73xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @MaxPoolNCEtoCMX
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x16x1x4xf16, {order = #NHWC}>)
func.func @MaxPoolNCEtoCMX(%arg0: tensor<1x16x1x4xf16, {order = #NHWC}>) -> tensor<1x16x1x4xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}> = dense<1> : tensor<16x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.MaxPool(%arg0, %weights) {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } -> tensor<1x16x1x4xf16, {order = #NHWC}>

    return %0 : tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_DDR:%.+]] = const.Declare tensor<16x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[IN_CMX:%.+]] = VPU.Copy([[ARG_0]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:       [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS_DDR]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<16x1x1x4xsi32, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.MaxPool([[IN_CMX]], [[WEIGHTS_CMX]] )
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[OUT_DDR:%.+]] = VPU.Copy([[OUT_CMX]])
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK:       return [[OUT_DDR]] : tensor<1x16x1x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @EltwiseAddNCEtoCMX
// CHECK-SAME: [[ARG_0:%[^:]+]]: tensor<1x64x28x28xf16, {order = #NHWC}>
// CHECK-SAME: [[ARG_1:%[^:]+]]: tensor<1x64x28x28xf16, {order = #NHWC}>
func.func @EltwiseAddNCEtoCMX(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>,
                         %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
                        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>
    } -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[IN1_CMX:%.+]] = VPU.Copy([[ARG_0]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
    // CHECK:       [[IN2_CMX:%.+]] = VPU.Copy([[ARG_1]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.Eltwise([[IN1_CMX]], [[IN2_CMX]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[OUT_DDR:%.+]] = VPU.Copy([[OUT_CMX]])
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[OUT_DDR]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @EltwiseAndSameInputsNCEtoCMX
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @EltwiseAndSameInputsNCEtoCMX(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>)
                                  -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
        op_type = #VPU.eltwise_type<AND>, ppe = #VPU.PPEStub<>
    } -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[IN_CMX:%.+]] = VPU.Copy([[ARG_0]]) {out_mem_space = [@CMX_NN, 0]}
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.Eltwise([[IN_CMX]], [[IN_CMX]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<AND>, ppe = #VPU.PPEStub<>
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[OUT_DDR:%.+]] = VPU.Copy([[OUT_CMX]])
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[OUT_DDR]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @InterpolateBilinearNCEtoCMX
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x64x5x10xf16, {order = #NHWC}>)
func.func @InterpolateBilinearNCEtoCMX(%arg0: tensor<1x64x5x10xf16, {order = #NHWC}>) -> tensor<1x64x10x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x2x2xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x2x2xf16>, [#const.Reorder<#NHWC>]
    %sparsityMap = const.Declare tensor<1x64x11x21xi1> = dense<1> : tensor<1x64x11x21xi1>

    %storageElement = VPU.StorageElementTable {
        dataElemType = f16,
        seDepth = 1, seSize = [64],
        dataShape = [1, 64, 5, 10],
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 11, 21]>
    } -> tensor<1x1x11x21xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsityMap, %storageElement) {
        seAttr = #VPU.SEInterpolate<
            mode = <BILINEAR>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 11, 21]>
    } ->
        !VPU.SparseTensor<
            data=tensor<1x64x5x10xf16, {order = #NHWC}>,
            sparsity_map=tensor<1x64x11x21xi1>,
            storage_element_table=tensor<1x1x11x21xi32, {order = #NHWC}>,
            #VPU.SEInterpolate<
                mode = <BILINEAR>,
                coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 2.0, 2.0],
                offsets = [0, 0, 0, 0],
                sizes = [1, 64, 11, 21]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights) rawFilterShape [64, 64, 2, 2] {
        strides = [1, 1],
        
        mode = #VPU.nce_interpolate_mode<BILINEAR>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x10x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x64x10x20xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<64x64x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x2x2xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[SPARSITY_MAP:%.+]] = const.Declare tensor<1x64x11x21xi1> = dense<true> : tensor<1x64x11x21xi1>

    // CHECK:       [[STORAGE_ELEMENT:%.+]] = VPU.StorageElementTable {
    // CHECK-SAME:      dataElemType = f16,
    // CHECK-SAME:      dataShape = [1, 64, 5, 10],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <BILINEAR>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 11, 21]>,
    // CHECK-SAME:      seDepth = 1 : i64,
    // CHECK-SAME:      seSize = [64]
    // CHECK-SAME:  } -> tensor<1x1x11x21xi32, {order = #NHWC}>

    // CHECK:       [[SPARSE_TENSOR:%.+]] = VPU.GroupSparseTensor([[ARG_0]], [[SPARSITY_MAP]], [[STORAGE_ELEMENT]]) {
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <BILINEAR>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 11, 21]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<
    // CHECK-SAME:      data=tensor<1x64x5x10xf16, {order = #NHWC}>,
    // CHECK-SAME:      sparsity_map=tensor<1x64x11x21xi1>,
    // CHECK-SAME:      storage_element_table=tensor<1x1x11x21xi32, {order = #NHWC}>,
    // CHECK-SAME:      #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <BILINEAR>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 11, 21]>>

    // CHECK:       [[COPY_0:%.+]] = VPU.Copy([[SPARSE_TENSOR]]) {out_mem_space = [@CMX_NN, 0]} : !VPU.SparseTensor<
    // CHECK-SAME:      data=tensor<1x64x5x10xf16, {order = #NHWC}>,
    // CHECK-SAME:      sparsity_map=tensor<1x64x11x21xi1>,
    // CHECK-SAME:      storage_element_table=tensor<1x1x11x21xi32, {order = #NHWC}>,
    // CHECK-SAME:      #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <BILINEAR>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 11, 21]>> ->
    // CHECK-SAME:  !VPU.SparseTensor<
    // CHECK-SAME:      data=tensor<1x64x5x10xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>,
    // CHECK-SAME:      sparsity_map=tensor<1x64x11x21xi1, {mem_space = [@CMX_NN, 0], order = #NCHW}>,
    // CHECK-SAME:      storage_element_table=tensor<1x1x11x21xi32, {mem_space = [@CMX_NN, 0], order = #NHWC}>,
    // CHECK-SAME:      #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <BILINEAR>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 11, 21]>>

    // CHECK:       [[COPY_1:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = [@CMX_NN, 0]} :
    // CHECK-SAME:      tensor<64x64x2x2xf16, {order = #NHWC}> -> tensor<64x64x2x2xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[INTERPOLATE:%.+]] = VPU.NCE.Interpolate([[COPY_0]], [[COPY_1]]) rawFilterShape [64, 64, 2, 2] {
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<BILINEAR>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      scales_attr = [2, 2]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } -> tensor<1x64x10x20xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[COPY_3:%.+]] = VPU.Copy([[INTERPOLATE]]) :
    // CHECK-SAME:      tensor<1x64x10x20xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x64x10x20xf16, {order = #NHWC}>

    // CHECK:       return [[COPY_3]] : tensor<1x64x10x20xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @InterpolateNearestNCEtoCMX
// CHECK-SAME: ([[ARG_0:%[^:]+]]: tensor<1x64x5x10xf16, {order = #NHWC}>)
func.func @InterpolateNearestNCEtoCMX(%arg0: tensor<1x64x5x10xf16, {order = #NHWC}>) -> tensor<1x64x10x20xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    %weightsTable = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    %sparsityMap = const.Declare tensor<1x64x10x20xi1> = dense<1> : tensor<1x64x10x20xi1>

    %storageElement = VPU.StorageElementTable {
        dataElemType = f16,
        seDepth = 1, seSize = [64],
        dataShape = [1, 64, 5, 10],
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 10, 20]>
    } -> tensor<1x1x10x20xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsityMap, %storageElement) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 64, 10, 20]>
    } ->
        !VPU.SparseTensor<
            data=tensor<1x64x5x10xf16, {order = #NHWC}>,
            sparsity_map=tensor<1x64x10x20xi1>,
            storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
            #VPU.SEInterpolate<
                mode = <NEAREST>,
                coordinate_transformation_mode = <ASYMMETRIC>,
                scale = [1.0, 1.0, 2.0, 2.0],
                nearest_mode = <FLOOR>,
                offsets = [0, 0, 0, 0],
                sizes = [1, 64, 10, 20]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights) rawFilterShape [64, 64, 1, 1] {
        strides = [1, 1],
        
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x64x10x20xf16, {order = #NHWC}>

    return %interpolate : tensor<1x64x10x20xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<64x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x64x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK:       [[SPARSITY_MAP:%.+]] = const.Declare tensor<1x64x10x20xi1> = dense<true> : tensor<1x64x10x20xi1>

    // CHECK:       [[STORAGE_ELEMENT:%.+]] = VPU.StorageElementTable {
    // CHECK-SAME:      dataElemType = f16,
    // CHECK-SAME:      dataShape = [1, 64, 5, 10],
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <NEAREST>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          nearest_mode = <FLOOR>,
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 10, 20]>,
    // CHECK-SAME:      seDepth = 1 : i64,
    // CHECK-SAME:      seSize = [64]
    // CHECK-SAME:  } -> tensor<1x1x10x20xi32, {order = #NHWC}>

    // CHECK:       [[SPARSE_TENSOR:%.+]] = VPU.GroupSparseTensor([[ARG_0]], [[SPARSITY_MAP]], [[STORAGE_ELEMENT]]) {
    // CHECK-SAME:      seAttr = #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <NEAREST>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          nearest_mode = <FLOOR>,
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 10, 20]>
    // CHECK-SAME:  } -> !VPU.SparseTensor<
    // CHECK-SAME:      data=tensor<1x64x5x10xf16, {order = #NHWC}>,
    // CHECK-SAME:      sparsity_map=tensor<1x64x10x20xi1>,
    // CHECK-SAME:      storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
    // CHECK-SAME:      #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <NEAREST>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          nearest_mode = <FLOOR>,
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 10, 20]>>

    // CHECK:       [[COPY_0:%.+]] = VPU.Copy([[SPARSE_TENSOR]]) {out_mem_space = [@CMX_NN, 0]} : !VPU.SparseTensor<
    // CHECK-SAME:      data=tensor<1x64x5x10xf16, {order = #NHWC}>,
    // CHECK-SAME:      sparsity_map=tensor<1x64x10x20xi1>,
    // CHECK-SAME:      storage_element_table=tensor<1x1x10x20xi32, {order = #NHWC}>,
    // CHECK-SAME:      #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <NEAREST>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          nearest_mode = <FLOOR>,
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 10, 20]>> ->
    // CHECK-SAME:  !VPU.SparseTensor<
    // CHECK-SAME:      data=tensor<1x64x5x10xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>,
    // CHECK-SAME:      sparsity_map=tensor<1x64x10x20xi1, {mem_space = [@CMX_NN, 0], order = #NCHW}>,
    // CHECK-SAME:      storage_element_table=tensor<1x1x10x20xi32, {mem_space = [@CMX_NN, 0], order = #NHWC}>,
    // CHECK-SAME:      #VPU.SEInterpolate<
    // CHECK-SAME:          mode = <NEAREST>,
    // CHECK-SAME:          coordinate_transformation_mode = <ASYMMETRIC>,
    // CHECK-SAME:          scale = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME:          nearest_mode = <FLOOR>,
    // CHECK-SAME:          offsets = [0, 0, 0, 0],
    // CHECK-SAME:          sizes = [1, 64, 10, 20]>>

    // CHECK:       [[COPY_1:%.+]] = VPU.Copy([[WEIGHTS]]) {out_mem_space = [@CMX_NN, 0]} :
    // CHECK-SAME:      tensor<64x64x1x1xf16, {order = #NHWC}> -> tensor<64x64x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[INTERPOLATE:%.+]] = VPU.NCE.Interpolate([[COPY_0]], [[COPY_1]]) rawFilterShape [64, 64, 1, 1] {
    // CHECK-SAME:      mode = #VPU.nce_interpolate_mode<NEAREST>,
    // CHECK-SAME:      ppe = #VPU.PPEStub<>,
    // CHECK-SAME:      scales_attr = [2, 2]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } -> tensor<1x64x10x20xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

    // CHECK:       [[COPY_3:%.+]] = VPU.Copy([[INTERPOLATE]]) :
    // CHECK-SAME:      tensor<1x64x10x20xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x64x10x20xf16, {order = #NHWC}>

    // CHECK:       return [[COPY_3]] : tensor<1x64x10x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0) -> (-d0 + 256, 48)>

//CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0) -> (-d0 + 256, 48)>

// CHECK-LABEL:   @AddCopiesInsideScfForAllOp
// CHECK-SAME:       [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @AddCopiesInsideScfForAllOp(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %0 = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
    %1 = scf.forall (%arg1) = (0) to (256) step (48) shared_outs(%arg2 = %0) -> (tensor<1x256x64x64xf16, {order = #NHWC}>) {
      %2 = affine.min #map(%arg1)
      %extracted_slice = tensor.extract_slice %cst[%arg1, 0, 0, 0] [%2, 32, 3, 3] [1, 1, 1, 1]
        : tensor<256x32x3x3xf16, {order = #NHWC}>
        to tensor<?x32x3x3xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

      %3 = VPU.NCE.Convolution(%arg0, %extracted_slice) rawFilterShape [256, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,  strides = [1, 1]
      } : tensor<1x32x64x64xf16, {order = #NHWC}>,
          tensor<?x32x3x3xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 3, 3]> : tensor<4xsi64>, order = #NHWC}>
          -> tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

      scf.forall.in_parallel {
        tensor.parallel_insert_slice %3 into %arg2[0, %arg1, 0, 0] [1, %2, 64, 64] [1, 1, 1, 1]
          : tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>
          into tensor<1x256x64x64xf16, {order = #NHWC}>
      }
    }
    return %1 : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK:   [[WEIGHTS_DDR:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}>

    // CHECK:   [[OUTPUT:%.+]] = tensor.empty() : tensor<1x256x64x64xf16, {order = #NHWC}>
    // CHECK:    scf.forall ([[LOOP_ITER:%.+]]) = (0) to (256) step (48) shared_outs([[LOOP_OUT:%.+]] = [[OUTPUT]])

    // CHECK:      [[SLICE_SIZE:%.+]] = affine.min #[[$MAP]]([[LOOP_ITER]])
    // CHECK:      [[WEIGHTS_SLICE:%.+]] = tensor.extract_slice [[WEIGHTS_DDR]][[[LOOP_ITER]], 0, 0, 0] [[[SLICE_SIZE]], 32, 3, 3]
    // CHECK-SAME:     : tensor<256x32x3x3xf16, {order = #NHWC}>
    // CHECK-SAME:     to tensor<?x32x3x3xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 3, 3]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:       [[IN_CMX:%.+]] = VPU.Copy([[INPUT]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<1x32x64x64xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:       [[WEIGHTS_CMX:%.+]] = VPU.Copy([[WEIGHTS_SLICE]]) {out_mem_space = @CMX_NN}
    // CHECK-SAME:      -> tensor<?x32x3x3xf16, {bounds = #const.OpaqueI64Elements<[256, 32, 3, 3]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[OUT_CMX:%.+]] = VPU.NCE.Convolution([[IN_CMX]], [[WEIGHTS_CMX]])
    // CHECK-SAME:      -> tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:       [[OUT_DDR:%.+]] = VPU.Copy([[OUT_CMX]])
    // CHECK-SAME:      -> tensor<1x?x64x64xf16, {bounds = #const.OpaqueI64Elements<[1, 256, 64, 64]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK:      tensor.parallel_insert_slice [[OUT_DDR]] into [[LOOP_OUT]][0, [[LOOP_ITER]], 0, 0] [1, [[SLICE_SIZE]], 64, 64] [1, 1, 1, 1]
}
