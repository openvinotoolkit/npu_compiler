//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --strategy-manager %s | FileCheck %s
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEAveragePoolOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x7x11520xf16, {order = #NHWC}>
func.func @SplitNCEAveragePoolOverW(%arg0: tensor<1x16x7x11520xf16, {order = #NHWC}>) -> tensor<1x16x1x11520xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {kernel_size = [7, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x16x1x11520xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x11520xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {kernel_size = [7, 1]
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 4]
    // CHECK-SAME:      -> tensor<1x16x1x11520xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x11520xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map < (d0, d1, d2, d3)->(d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOH
func.func @ConvAssignedSOH(%arg0 : tensor<1x64x28x28xf16, {order = #NHWC}>)->tensor<1x80x28x28xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>

    %cst_0 = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>,[#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding< left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64 >,
        rawFilterShape = [ 80, 64, 3, 3 ], strides = [ 1, 1 ]
    } ->tensor<1x80x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x80x28x28xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<80x1x1x4xsi32> = dense<10> : tensor<80x1x1x4xsi32>
    // CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<80x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<80x64x3x3xf16>, [#const.Reorder<#NHWC>]

    // CHECK:        [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0, %cst_0, %cst)
    // CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

    // CHECK:        return [[VAL0]] : tensor<1x80x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map < (d0, d1, d2, d3)->(d0, d2, d3, d1)>

// CHECK-LABEL: @ConvAssignedSOK
func.func @ConvAssignedSOK(%arg0 : tensor<1x64x1x1xf16, {order = #NHWC}>)->tensor<1x48x1x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    %cst_0 = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %cst_0, %cst) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding< left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64 >,
        rawFilterShape = [ 48, 64, 3, 3 ], strides = [ 1, 1 ]
    } -> tensor<1x48x1x1xf16, {order = #NHWC}>

    return %0 : tensor<1x48x1x1xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<48x1x1x4xsi32> = dense<10> : tensor<48x1x1x4xsi32>
    // CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<48x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x3x3xf16>, [#const.Reorder<#NHWC>]

    // CHECK:        [[VAL0:%.*]] = VPU.NCE.Convolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    // CHECK-SAME:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:   -> tensor<1x48x1x1xf16, {order = #NHWC}>

    // CHECK:        return [[VAL0]] : tensor<1x48x1x1xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map < (d0, d1, d2, d3)->(d0, d1, d2, d3)>
#NHWC = affine_map < (d0, d1, d2, d3)->(d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvAssignedSOK
func.func @DepthConvAssignedSOK(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>) -> tensor<1x32x112x112xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [
          #const.Reshape<[ 32, 1, 3, 3 ]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>,
          #const.Reshape<[ 32, 9, 1, 1 ]>, #const.PadWithZero<[ 0, 0, 0, 0 ], [ 0, 7, 0, 0 ]>, #const.Reorder<#NHWC>
      ]

    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding< left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64 >,
        rawFilterShape = [ 32, 1, 3, 3 ], strides = [ 1, 1 ]
    } -> tensor<1x32x112x112xf16, {order = #NHWC}>

    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    // CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:   = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.Reshape<[32, 1, 3, 3]>,
    // #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0,
    // 7, 0, 0]>, #const.Reorder<#NHWC>]

    // CHECK:        [[VAL0:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    // CHECK-SAME:  #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME: -> tensor<1x32x112x112xf16, {order = #NHWC}>

    // CHECK:        return [[VAL0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map < (d0, d1, d2, d3)->(d0, d2, d3, d1)>
#NCHW = affine_map < (d0, d1, d2, d3)->(d0, d1, d2, d3)>

// CHECK-LABEL: @DepthConvAssignedSOK
func.func @DepthConvAssignedSOK(%arg0: tensor<1x128x1x1xf16, {order = #NHWC}>) -> tensor<1x128x1x1xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>

    %cst_1 = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<128x1x1x3x3xf16>,
      [
          #const.Reshape<[ 128, 1, 3, 3 ]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>,
          #const.Reshape<[ 128, 9, 1, 1 ]>, #const.PadWithZero<[ 0, 0, 0, 0 ], [ 0, 7, 0, 0 ]>, #const.Reorder<#NHWC>
      ]

    %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding< left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64 >,
        rawFilterShape = [ 128, 1, 3, 3 ], strides = [ 1, 1 ]
    } -> tensor<1x128x1x1xf16, {order = #NHWC}>

    return %0 : tensor<1x128x1x1xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<128x1x1x4xsi32> = dense<10> : tensor<128x1x1x4xsi32>
    // CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<128x16x1x1xf16, {order = #NHWC}>

    // CHECK:        [[VAL0:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    // CHECK-SAME:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>

    // CHECK:        return [[VAL0]] : tensor<1x128x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map < (d0, d1, d2, d3)->(d0, d2, d3, d1)>
#NCHW = affine_map < (d0, d1, d2, d3)->(d0, d1, d2, d3)>

// CHECK-LABEL: @DepthConvAssignedSOK
func.func @DepthConvAssignedSOK(%arg0 : tensor<1x32x1x1xf16, {order = #NHWC}>)->tensor<1x32x1x1xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>

    %cst_1 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>,
      [
          #const.Reshape<[ 32, 1, 3, 3 ]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>,
          #const.Reshape<[ 32, 9, 1, 1 ]>, #const.PadWithZero<[ 0, 0, 0, 0 ], [ 0, 7, 0, 0 ]>, #const.Reorder<#NHWC>
      ]

     %0 = VPU.NCE.DepthConvolution(%arg0, %cst_1, %cst_0) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding< left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64 >,
        rawFilterShape = [ 32, 1, 3, 3 ], strides = [ 1, 1 ]
    } -> tensor<1x32x1x1xf16, {order = #NHWC}>

    return %0 : tensor<1x32x1x1xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTSTABLE:%.*]] = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    // CHECK:        [[WEIGHTS:%.*]] = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}>
    // CHECK-SAME:   = dense<1.000000e+00> : tensor<32x1x1x3x3xf16>, [#const.Reshape<[32, 1, 3, 3]>,
    // #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[32, 9, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0,
    // 7, 0, 0]>, #const.Reorder<#NHWC>]

    // CHECK:        [[VAL0:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[WEIGHTS]], [[WEIGHTSTABLE]])
    // CHECK-SAME:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>

    // CHECK:        return [[VAL0]] : tensor<1x32x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map < (d0, d1, d2, d3)->(d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolAssignedSOH
func.func @MaxPoolAssignedSOH(%arg0: tensor<1x32x112x112xf16, {order = #NHWC}>)->tensor<1x32x112x112xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding< left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64 >,
        strides = [ 1, 1 ], kernel_size = [ 1, 1 ]
    }->tensor<1x32x112x112xf16, {order = #NHWC}>

    return %0 : tensor<1x32x112x112xf16, {order = #NHWC}>

    // CHECK:        [[VAL0:%.*]] = VPU.NCE.MaxPool(%arg0)
    // CHECK-SAME:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

    // CHECK:        return [[VAL0]] : tensor<1x32x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map < (d0, d1, d2, d3)->(d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolAssignedSOK
func.func @MaxPoolAssignedSOK(%arg0: tensor<1x32x1x1xf16, {order = #NHWC}>)->tensor<1x32x1x1xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding< left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64 >,
        strides = [ 1, 1 ], kernel_size = [ 1, 1 ]
    }->tensor<1x32x1x1xf16, {order = #NHWC}>

   return %0 : tensor<1x32x1x1xf16, {order = #NHWC}>

    // CHECK:        [[VAL0:%.*]] = VPU.NCE.MaxPool(%arg0)
    // CHECK-SAME:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>

    // CHECK:        return [[VAL0]] : tensor<1x32x1x1xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map < (d0, d1, d2, d3)->(d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddAssignedSOH
func.func @EltwiseAddAssignedSOH(%arg0
                                 : tensor<1x32x80x80xf16, {order = #NHWC}>, %arg1
                                 : tensor<1x32x80x80xf16, {order = #NHWC}>)
        ->tensor<1x32x80x80xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1){op_type = #VPU.eltwise_type<ADD>,ppe = #VPU.PPEStub<>} : tensor<1x32x80x80xf16, {order = #NHWC}>,
      tensor<1x32x80x80xf16, {order = #NHWC}>->tensor<1x32x80x80xf16, {order = #NHWC}>
return %0 : tensor<1x32x80x80xf16, {order = #NHWC}>

    // CHECK:      [[VAL0:%.*]] = VPU.NCE.Eltwise(%arg0, %arg1)
    // CHECK-SAME: {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>

    // CHECK:        return [[VAL0]] : tensor<1x32x80x80xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map < (d0, d1, d2, d3)->(d0, d1, d2, d3)>

// CHECK-LABEL: @TanhAssignedClustering
func.func @TanhAssignedClustering(%arg0: tensor<1x4x1x512xf16, {order = #NCHW}>)->tensor<1x4x1x512xf16, {order = #NCHW}> {
    %1 = VPU.Tanh(%arg0) : tensor<1x4x1x512xf16, {order = #NCHW}>->tensor<1x4x1x512xf16, {order = #NCHW}>

    return %1 : tensor<1x4x1x512xf16, {order = #NCHW}>

    // CHECK:   [[ResultTANH:%.*]] = VPU.Tanh(%arg0)
    // CHECK-SAME: {multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>}
}

// -----

#NCHW = affine_map < (d0, d1, d2, d3)->(d0, d1, d2, d3)>

// CHECK-LABEL: @MVNAssignedClustering
func.func @MVNAssignedClustering(%arg0 : tensor<1x1x1x512xf16, {order = #NCHW}>)->tensor<1x1x1x512xf16, {order = #NCHW}> {
    %0 = VPU.MVN(%arg0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, normalize_variance = true} : tensor<1x1x1x512xf16, {order = #NCHW}> -> tensor<1x1x1x512xf16, {order = #NCHW}>

    return %0 : tensor<1x1x1x512xf16, {order = #NCHW}>

    // CHECK:   [[ResultMVN:%.*]] = VPU.MVN(%arg0)
    // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    // CHECK:   return [[ResultMVN]] : tensor<1x1x1x512xf16, {order = #NCHW}>
}

// -----

#NCHW = affine_map < (d0, d1, d2, d3)->(d0, d1, d2, d3)>

// CHECK-LABEL: @SoftMaxAssignedClustering
func.func @SoftMaxAssignedClustering(%arg0 : tensor<1x4x1x512xf16, {order = #NCHW}>)->tensor<1x4x1x512xf16, {order = #NCHW}> {
    %1 = VPU.SoftMax(%arg0) {axisInd = 1} : tensor<1x4x1x512xf16, {order = #NCHW}> -> tensor<1x4x1x512xf16, {order = #NCHW}>

    return %1 : tensor<1x4x1x512xf16, {order = #NCHW}>

    // CHECK:   [[ResultSoftMax:%.*]] = VPU.SoftMax(%arg0)
    // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    // CHECK:   return [[ResultSoftMax]] : tensor<1x4x1x512xf16, {order = #NCHW}>
}

// -----

// CHECK-LABEL: @InterpolateHalfPixelAssignedSOHOverlapped
func.func @InterpolateHalfPixelAssignedSOHOverlapped(%arg0: tensor<1x1x96x160xf16>) -> tensor<1x1x192x320xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320]} : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
    return %0 : tensor<1x1x192x320xf16>
    // CHECK:   [[INTERPOLATE:%.*]] = VPU.Interpolate(%arg0)
    // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>
    // CHECK:   return [[INTERPOLATE]] : tensor<1x1x192x320xf16>
}

// -----

// CHECK-LABEL: @InterpolateAlignCornersAssignedSOHOverlapped
func.func @InterpolateAlignCornersAssignedSOHOverlapped(%arg0: tensor<1x1x96x160xf16>) -> tensor<1x1x192x320xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320]} : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
    return %0 : tensor<1x1x192x320xf16>
    // CHECK:   [[INTERPOLATE:%.*]] = VPU.Interpolate(%arg0)
    // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>
    // CHECK:   return [[INTERPOLATE]] : tensor<1x1x192x320xf16>
}

// -----

// CHECK-LABEL: @InterpolateAlignCornersAssignedClustering
func.func @InterpolateAlignCornersAssignedClustering(%arg0: tensor<1x1x1x160xf16>) -> tensor<1x1x1x320xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [1.000000e+00, 2.000000e+00], sizes_attr = [1, 320]} : tensor<1x1x1x160xf16> -> tensor<1x1x1x320xf16>
    return %0 : tensor<1x1x1x320xf16>
    // CHECK:   [[INTERPOLATE:%.*]] = VPU.Interpolate(%arg0)
    // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    // CHECK:   return [[INTERPOLATE]] : tensor<1x1x1x320xf16>
}

// -----

// CHECK-LABEL: @InterpolatePytorchHalfPixelAssignedSOHOverlapped
func.func @InterpolatePytorchHalfPixelAssignedSOHOverlapped(%arg0: tensor<1x1x96x160xf16>) -> tensor<1x1x192x320xf16> {
    %0 = VPU.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <PYTORCH_HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 320]} : tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
    return %0 : tensor<1x1x192x320xf16>
    // CHECK:   [[INTERPOLATE:%.*]] = VPU.Interpolate(%arg0)
    // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>
    // CHECK-SAME: tensor<1x1x96x160xf16> -> tensor<1x1x192x320xf16>
    // CHECK:   return [[INTERPOLATE]] : tensor<1x1x192x320xf16>
}

// -----

#NHWC = affine_map < (d0, d1, d2, d3)->(d0, d2, d3, d1)>

// CHECK-LABEL: @CompressConvolutionAssignedSOHOverlapped
func.func @CompressConvolutionAssignedSOHOverlapped(%arg0 : tensor<1x3x224x224xf16, {order = #NHWC}>)->tensor<1x64x112x112xf16, {order = #NHWC}> {
    %weight_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %filter = const.Declare tensor<64x1x1x160xf16, {order = #NHWC}> = dense<1.0> : tensor<64x3x7x7xf16>, [#const.CastElemType<ui8>,
            #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>, #const.SubView<[0, 0, 0, 0], [64, 3, 7, 7]>,
            #const.Reshape<[64, 1, 1, 147]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 13]>]

    %expand = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]}
            : tensor<1x3x224x224xf16, {order = #NHWC}> -> tensor<1x4x224x224xf16, {order = #NHWC}>

    %compress_conv = VPU.NCE.CompressConvolution(%expand, %filter, %weight_table)
            {
        cm_sp_pattern = 15 : i64, pad = #VPU.Padding< left = 3 : i64, right = 2 : i64, top = 3 : i64,
        bottom = 2 : i64 >, ppe = #VPU.PPEStub<>,
        rawFilterShape = [ 64, 4, 7, 7 ], strides = [ 2, 2 ]
            } -> tensor<1x64x112x112xf16, {order = #NHWC}>

    return %compress_conv : tensor<1x64x112x112xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>
    // CHECK:        [[FILTER:%.+]] = const.Declare tensor<64x1x1x160xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x3x7x7xf16>
    // CHECK:        [[EXPAND:%.+]] = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 1, 0, 0]} :
    // CHECK-SAME:           tensor<1x3x224x224xf16, {order = #NHWC}> -> tensor<1x4x224x224xf16, {order = #NHWC}>

    // CHECK:        [[VAL0:%.+]] = VPU.NCE.CompressConvolution([[EXPAND]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:   multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>
    // CHECK-SAME:   tensor<1x64x112x112xf16, {order = #NHWC}>

    // CHECK:        return [[VAL0]] : tensor<1x64x112x112xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map < (d0, d1, d2, d3)->(d0, d2, d3, d1)>

// CHECK-LABEL: @MultiplyAssignedClustering
func.func @MultiplyAssignedClustering(%arg0 : tensor<1x1x1x44xf16, {order = #NHWC}>,
                                      %arg1 : tensor<1x1x1x1xf16, {order = #NHWC}>)->tensor<1x1x1x44xf16, {order = #NHWC}> {
    %0 = VPU.Multiply(%arg0, %arg1){auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
            tensor<1x1x1x44xf16, {order = #NHWC}>,
      tensor<1x1x1x1xf16, {order = #NHWC}>->tensor<1x1x1x44xf16, {order = #NHWC}>

    return %0 : tensor<1x1x1x44xf16, {order = #NHWC}>

    // CHECK:      [[MULTIPLY:%.*]] = VPU.Multiply(%arg0, %arg1)
    // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    // CHECK:      return [[MULTIPLY]] : tensor<1x1x1x44xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map < (d0, d1, d2, d3)->(d0, d1, d2, d3)>

// CHECK-LABEL: @HSwishAssignedSplitOverKernel
func.func @HSwishAssignedSplitOverKernel(%arg0
                                         : tensor<1x4x1x512xf16, {order = #NCHW}>)->tensor<1x4x1x512xf16, {order = #NCHW}> {
    %0 = VPU.HSwish(%arg0)
            : tensor<1x4x1x512xf16, {order = #NCHW}>->tensor<1x4x1x512xf16, {order = #NCHW}>

    return %0 : tensor<1x4x1x512xf16, {order = #NCHW}>

    // CHECK:   [[ResultHSwish:%.*]] = VPU.HSwish(%arg0)
    // CHECK-SAME: multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>}
    // CHECK:   return [[ResultHSwish]] : tensor<1x4x1x512xf16, {order = #NCHW}>
}

// -----

#NHWC = affine_map < (d0, d1, d2, d3)->(d0, d2, d3, d1)>

// CHECK-LABEL: @InterpolateNearestAssignedClustering
func.func @InterpolateNearestAssignedClustering(%arg0: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %sparsity_map = const.Declare tensor<1x16x2x2xi1> = dense<1> : tensor<1x16x2x2xi1>

    %storage_element = VPU.StorageElementTable {dataElemType = i32, seDepth = 1, seSize = 16, dataShape = [1, 16, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>
    } -> tensor<1x1x2x2xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 16, 2, 2]>
    } -> !VPU.SparseTensor<data=tensor<1x16x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x16x2x2xi1>,
                           storage_element_table=tensor<1x1x2x2xi32, {order = #NHWC}>,
       #VPU.SEInterpolate<mode = < NEAREST>, coordinate_transformation_mode = < ASYMMETRIC>,
       scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 16, 2, 2]>>

    %interpolate = VPU.NCE.Interpolate(%input, %weights, %weights_table) {
        rawFilterShape = [16, 16, 1, 1],
        strides = [1, 1],
        mode = #VPU.nce_interpolate_mode<NEAREST>,
        scales_attr = [2, 2],
        ppe = #VPU.PPEStub<>
    } -> tensor<1x16x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x16x2x2xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK-DAG:   [[INPUT_SM:%.+]] = const.Declare tensor<1x16x2x2xi1> = dense<true> : tensor<1x16x2x2xi1>

    // CHECK:       [[INPUT_SE:%.+]] = VPU.StorageElementTable
    // CHECK:       [[INPUT_SPARSE:%.+]] = VPU.GroupSparseTensor(%arg0, [[INPUT_SM]], [[INPUT_SE]])

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Interpolate([[INPUT_SPARSE]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>,
    // CHECK:       return [[OUTPUT]] : tensor<1x16x2x2xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL:   func.func @ConvertOpAssignedMCStratF32To16
func.func @ConvertOpAssignedMCStratF32To16(%arg0: tensor<1x48x160x80xf32>) -> tensor<1x48x160x80xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    return %0 : tensor<1x48x160x80xf16>
    // CHECK:       [[CONVERT:%.+]] = VPU.Convert(%arg0) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    // CHECK:       return [[CONVERT]] : tensor<1x48x160x80xf16>
}

// -----

// CHECK-LABEL:   func.func @ConvertOpAssignedMCStratF16To32
func.func @ConvertOpAssignedMCStratF16To32(%arg0: tensor<1x48x160x80xf16>) -> tensor<1x48x160x80xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x48x160x80xf16> -> tensor<1x48x160x80xf32>
    return %0 : tensor<1x48x160x80xf32>
    // CHECK:       [[CONVERT:%.+]] = VPU.Convert(%arg0) {dstElemType = f32, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x48x160x80xf16> -> tensor<1x48x160x80xf32>
    // CHECK:       return [[CONVERT]] : tensor<1x48x160x80xf32>
}

// -----

// CHECK-LABEL:   func.func @ConvertOpAssignedMCStratClustering
func.func @ConvertOpAssignedMCStratClustering(%arg0: tensor<2x48x160x80xf16>) -> tensor<2x48x160x80xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<2x48x160x80xf16> -> tensor<2x48x160x80xf32>
    return %0 : tensor<2x48x160x80xf32>
    // CHECK:       [[CONVERT:%.+]] = VPU.Convert(%arg0) {dstElemType = f32,
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    // CHECK-SAME:  tilingStrategy = [1, 1, 4, 1]}
    // CHECK-SAME:  tensor<2x48x160x80xf16> -> tensor<2x48x160x80xf32>
    // CHECK:       return [[CONVERT]] : tensor<2x48x160x80xf32>
}

// -----

// CHECK-LABEL:   func.func @ConvertOpAssignedMCStratSOK
func.func @ConvertOpAssignedMCStratSOK(%arg0: tensor<1x48x3x3xf16>) -> tensor<1x48x3x3xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x48x3x3xf16> -> tensor<1x48x3x3xf32>
    return %0 : tensor<1x48x3x3xf32>
    // CHECK:       [[CONVERT:%.+]] = VPU.Convert(%arg0) {dstElemType = f32, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x48x3x3xf16> -> tensor<1x48x3x3xf32>
    // CHECK:       return [[CONVERT]] : tensor<1x48x3x3xf32>
}

// -----

// CHECK-LABEL: @AssignSOHForLayerWithTopK
func.func @AssignSOHForLayerWithTopK(%arg0: tensor<1x151x513x513xf16>) -> tensor<1x1x513x513xsi32> {
    %cst = const.Declare tensor<1x1x1x16xui8> = dense<0> : tensor<1x1x1x16xui8>
    %output_values, %target_shape = VPU.TopK(%arg0, %cst)
        {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, operandSegmentSizes = array<i32: 1, 0, 1>,  sort = #IE.topk_sort_type<SORT_INDICES>}
            : tensor<1x151x513x513xf16>, tensor<1x1x1x16xui8> -> tensor<1x1x513x513xf16>, tensor<1x1x513x513xsi32>
    return %target_shape : tensor<1x1x513x513xsi32>

    // CHECK:        [[CST:%.+]] = const.Declare tensor<1x1x1x16xui8> = dense<0> : tensor<1x1x1x16xui8>
    // CHECK:        [[OUTPUT_VALUES:%.+]], [[TARGET_SHAPE:%.+]] = VPU.TopK(%arg0, [[CST]])
    // CHECK-SAME:        axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>,
    // CHECK-SAME:        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, operandSegmentSizes = array<i32: 1, 0, 1>, sort = #IE.topk_sort_type<SORT_INDICES>
    // CHECK-SAME:        tilingStrategy = [1, 1, 43, 1]
    // CHECK-SAME:        : tensor<1x151x513x513xf16>, tensor<1x1x1x16xui8> -> tensor<1x1x513x513xf16>, tensor<1x1x513x513xsi32> 
    // CHECK:        return [[TARGET_SHAPE]] : tensor<1x1x513x513xsi32>
}

// -----

#NHWC = affine_map < (d0, d1, d2, d3)->(d0, d2, d3, d1)>

// CHECK-LABEL: @TileWithSOKTiling
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x30x30xf16, {order = #NHWC}>
func.func @TileWithSOKTiling(%arg0 : tensor<1x32x30x30xf16, {order = #NHWC}>)->tensor<1x768x30x30xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<768x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<768x32x7x7xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<768x1x1x4xsi32> = dense<1> :
            tensor<768x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
        rawFilterShape = [ 768, 32, 7, 7 ], strides = [ 1, 1 ]
    } ->tensor<1x768x30x30xf16, {order = #NHWC}>

    return %0 : tensor<1x768x30x30xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<768x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:          : tensor<768x32x7x7xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<768x1x1x4xsi32> = dense<1>
    // CHECK-SAME:          : tensor<768x1x1x4xsi32>

    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:          pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>
    // CHECK-SAME:          rawFilterShape = [768, 32, 7, 7],
    // CHECK-SAME:          strides = [1, 1],
    // CHECK-SAME:          tilingStrategy = [1, 4, 1, 1]}
    // CHECK-SAME:        -> tensor<1x768x30x30xf16, {order = #NHWC}>

    // CHECK:       return [[CONV1]] : tensor<1x768x30x30xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.15407909318512561:127>

// -----

!qElemType = !quant.uniform<u8:f16, 0.15407909318512561:127>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @LastOpExceptionCase
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x60x60x!qElemType, {order = #NHWC}>
func.func @LastOpExceptionCase(%arg0: tensor<1x32x60x60x!qElemType, {order = #NHWC}>) -> tensor<1x32x15x15x!qElemType, {order = #NHWC}>  {
    %cst = const.Declare tensor<16x32x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x3x3xf32, {order = #NHWC}>
    %cst_0 = const.Declare tensor<64x16x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x16x3x3xf32, {order = #NHWC}>
    %cst_1 = const.Declare tensor<32x64x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x64x3x3xf32, {order = #NHWC}>
    %cst_2 = const.Declare tensor<32x1x1x4xsi32> = dense<10> : tensor<32x1x1x4xsi32>
    %cst_3 = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %cst_4 = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense<10.0> : tensor<1x64x1x1xf32>, [#const.CastElemType<f16>, #const.Reshape<[64, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Reorder<#NCHW>, #const.Reshape<[64, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_5 = const.Declare tensor<64x1x1x4xsi32> = dense<10> : tensor<64x1x1x4xsi32>
    %cst_6 = const.Declare tensor<16x1x1x4xsi32> = dense<10> : tensor<16x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %cst, %cst_6) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [16, 32, 3, 3], strides = [2, 2]} -> tensor<1x16x30x30x!qElemType, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst_0, %cst_5) {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 16, 3, 3], strides = [1, 1]} -> tensor<1x64x30x30xf16, {order = #NHWC}>
    %2 = VPU.NCE.DepthConvolution(%1, %cst_4, %cst_3) {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [64, 1, 1, 1], strides = [1, 1]} -> tensor<1x64x30x30x!qElemType, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_1, %cst_2) {pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [32, 64, 3, 3], strides = [2, 2]} -> tensor<1x32x15x15x!qElemType, {order = #NHWC}>

    return %3 : tensor<1x32x15x15x!qElemType, {order = #NHWC}>
    // CHECK-COUNT-7:    const.Declare
    //CHECK:            [[VAL0:%.+]] = VPU.NCE.Convolution(%arg0,
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>
    //CHECK:            [[VAL1:%.+]] = VPU.NCE.Convolution([[VAL0]],
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    //CHECK:            [[VAL2:%.+]] = VPU.NCE.DepthConvolution([[VAL1]],
    //CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    //CHECK:            [[VAL3:%.+]] = VPU.NCE.Convolution([[VAL2]],
    //CHECK-SAME:         multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    //CHECK-NOT:          multiClusterStrategy = #VPU.multi_cluster_strategy<HKSwitch>

    //CHECK:            return [[VAL3]] : tensor<1x32x15x15x!qElemType, {order = #NHWC}>

}
