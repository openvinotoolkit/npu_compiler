//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --tiling-strategy-assignment="tiling-mode=ISOLATED enable-shave-ddr-access-optimization=true" %s | FileCheck %s
// REQUIRES: arch-NPU37XX

// CHECK-LABEL: func.func @SplitSwConvOverOC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<256x32x3x3xf16>,
// CHECK-SAME:        [[BIAS:%arg[0-9]]]: tensor<1x256x1x1xf16>
func.func @SplitSwConvOverOC(
        %input: tensor<1x32x64x64xf16>,
        %filter: tensor<256x32x3x3xf16>,
        %bias: tensor<1x256x1x1xf16>)
            -> tensor<1x256x64x64xf16> {
    %1 = VPU.Convolution(%input, %filter, %bias) {
        ppe = #VPU.PPEStub<>,
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x32x64x64xf16>, tensor<256x32x3x3xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x64x64xf16>
    return %1 : tensor<1x256x64x64xf16>

    // CHECK:       [[CONV:%.+]] = VPU.Convolution([[INPUT]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      -> tensor<1x256x64x64xf16>

    // CHECK:       return [[CONV]] : tensor<1x256x64x64xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoSplitDepthConvOverCWithSOK
func.func @NoSplitDepthConvOverCWithSOK(%arg0: tensor<1x80x3840x4xf16, {order = #NHWC}>) -> tensor<1x80x3840x4xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<80x16x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<80x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<80x1x1x4xsi32, {order = #NHWC}> =
        dense<10> : tensor<80x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.DepthConvolution(%arg0, %weights, %wt) {
            ppe = #VPU.PPEStub<>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [80, 1, 1, 1], strides = [1, 1]
        } -> tensor<1x80x3840x4xf16, {order = #NHWC}>

    return %0 : tensor<1x80x3840x4xf16, {order = #NHWC}>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<80x16x1x1xf16, {order = #NHWC}>
    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<80x1x1x4xsi32, {order = #NHWC}>
    // CHECK: [[DWConv:%.*]] = VPU.NCE.DepthConvolution(%arg0, [[CST]], [[CST0]])
    // CHECK-SAME:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
    // CHECK-SAME:              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:               rawFilterShape = [80, 1, 1, 1], strides = [1, 1],
    // CHECK-SAME:               tilingStrategy = [1, 1, 2, 1]} -> tensor<1x80x3840x4xf16, {order = #NHWC}>
    // CHECK:  return [[DWConv]] : tensor<1x80x3840x4xf16, {order = #NHWC}>
}

// -----
// CHECK-LABEL: func.func @SplitSwMaxPoolOverH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x200x200xf16>
func.func @SplitSwMaxPoolOverH(
        %input: tensor<1x16x200x200xf16>)
            -> tensor<1x16x200x200xf16> {
    %1 = VPU.MaxPool(%input) {
        ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x200x200xf16> -> tensor<1x16x200x200xf16>
    return %1 : tensor<1x16x200x200xf16>

    // CHECK:       [[MAXPOOL:%.+]] = VPU.MaxPool([[INPUT]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:      -> tensor<1x16x200x200xf16>

    // CHECK:       return [[MAXPOOL]] : tensor<1x16x200x200xf16>
}

// -----

// CHECK-LABEL: func.func @SplitSwAddOverC
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x2048x14x14xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x2048x14x14xf16>
func.func @SplitSwAddOverC(
        %input1: tensor<1x2048x14x14xf16>,
        %input2: tensor<1x2048x14x14xf16>)
            -> tensor<1x2048x14x14xf16> {
    %1 = VPU.Add(%input1, %input2) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x2048x14x14xf16>, tensor<1x2048x14x14xf16> -> tensor<1x2048x14x14xf16>
    return %1 : tensor<1x2048x14x14xf16>

    // CHECK:       [[ADD:%.+]] = VPU.Add([[INPUT1]], [[INPUT2]])
    // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      -> tensor<1x2048x14x14xf16>

    // CHECK:       return [[ADD]] : tensor<1x2048x14x14xf16>
}

// -----

// CHECK-LABEL: func.func @SplitAddSameInputOverC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x2048x14x14xf16>
func.func @SplitAddSameInputOverC(
        %input: tensor<1x2048x14x14xf16>)
            -> tensor<1x2048x14x14xf16> {
    %1 = VPU.Add(%input, %input) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x2048x14x14xf16>, tensor<1x2048x14x14xf16> -> tensor<1x2048x14x14xf16>
    return %1 : tensor<1x2048x14x14xf16>

    // CHECK:       [[AND:%.+]] = VPU.Add([[INPUT]], [[INPUT]])
    // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      -> tensor<1x2048x14x14xf16>

    // CHECK:       return [[AND]] : tensor<1x2048x14x14xf16>
}

// -----

// CHECK-LABEL: func.func @InterpSplitOverC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16>
func.func @InterpSplitOverC(
        %input1: tensor<1x32x64x64xf16>)
            -> tensor<1x32x256x256xf16> {

    %0 = VPU.Interpolate(%input1) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <LINEAR>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
            axes_attr = [2, 3], sizes_attr = [256, 256], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0> } :
        tensor<1x32x64x64xf16> -> tensor<1x32x256x256xf16>

    return %0 : tensor<1x32x256x256xf16>

    // CHECK:       [[INTERP0:%.+]] = VPU.Interpolate([[INPUT]]
    // CHECK-SAME:      tilingStrategy = [1, 1, 3, 1]
    // CHECK-SAME:      : tensor<1x32x64x64xf16>
    // CHECK-SAME:      -> tensor<1x32x256x256xf16>

    // CHECK:       return [[INTERP0]] : tensor<1x32x256x256xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @InterpSplitOverH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x64x48x80xf16, {order = #NHWC}>
func.func @InterpSplitOverH(
    %arg0: tensor<1x64x48x80xf16, {order = #NHWC}>)
            -> tensor<1x64x192x320xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode =  <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        sizes_attr = [192, 320],
        tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} :
        tensor<1x64x48x80xf16, {order = #NHWC}> -> tensor<1x64x192x320xf16, {order = #NHWC}>
    return %0 : tensor<1x64x192x320xf16, {order = #NHWC}>

    // CHECK:  [[INTERP0:%.+]] = VPU.Interpolate(%arg0)
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 5]
    // CHECH-SAME:  : tensor<1x64x48x80xf16, {order = #NHWC}>
    // CHECH-SAME:  -> tensor<1x64x192x320xf16, {order = #NHWC}>

    // CHECK:  return [[INTERP0]] : tensor<1x64x192x320xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @InterpSplitOverHW
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x128x35x35xf16, {order = #NHWC}>
func.func @InterpSplitOverHW(
    %input1: tensor<1x128x35x35xf16, {order = #NHWC}>)
            -> tensor<1x128x168x335xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%input1) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <LINEAR>, nearest_mode =  <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        sizes_attr = [168, 335]} :
        tensor<1x128x35x35xf16, {order = #NHWC}> -> tensor<1x128x168x335xf16, {order = #NHWC}>
    return %0 : tensor<1x128x168x335xf16, {order = #NHWC}>

    // CHECK:  [[INTERP0:%.+]] = VPU.Interpolate(%arg0)
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 8]
    // CHECH-SAME:  : tensor<1x128x35x35xf16, {order = #NHWC}>
    // CHECH-SAME:  -> tensor<1x128x168x335xf16, {order = #NHWC}>

    // CHECK:  return [[INTERP0]] : tensor<1x128x168x335xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @InterpSplitOverCNoCommonFactor
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x64x31x31xf16, {order = #NHWC}>
func.func @InterpSplitOverCNoCommonFactor(
    %arg0: tensor<1x64x31x31xf16, {order = #NHWC}>)
            -> tensor<1x64x121x121xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode =  <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        sizes_attr = [121, 121],
        tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} :
        tensor<1x64x31x31xf16, {order = #NHWC}> -> tensor<1x64x121x121xf16, {order = #NHWC}>
    return %0 : tensor<1x64x121x121xf16, {order = #NHWC}>

    // CHECK:  [[INTERP0:%.+]] = VPU.Interpolate(%arg0)
    // CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]
    // CHECH-SAME:  : tensor<1x64x31x31xf16, {order = #NHWC}>
    // CHECH-SAME:  -> tensor<1x64x121x121xf16, {order = #NHWC}>

    // CHECK:  return [[INTERP0]] : tensor<1x64x121x121xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: func.func @MaximumSplitOverH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x2048x2048xf16>
func.func @MaximumSplitOverH(
    %arg0: tensor<1x32x2048x2048xf16>)
            -> tensor<1x32x2048x2048xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %0 = VPU.Maximum(%arg0, %cst) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
        tensor<1x32x2048x2048xf16>, tensor<1x1x1x1xf16>
        -> tensor<1x32x2048x2048xf16>

    return %0 : tensor<1x32x2048x2048xf16>

    // CHECK:  [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:  [[MAXIMUM0:%.+]] = VPU.Maximum(%arg0, [[CST]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:  tilingStrategy = [1, 1, 147, 1]
    // CHECH-SAME:  : tensor<1x32x2048x2048xf16>, tensor<1x1x1x1xf16>
    // CHECH-SAME:  -> tensor<1x32x2048x2048xf16>

    // CHECK:  return [[MAXIMUM0]] : tensor<1x32x2048x2048xf16>
}

// -----

// CHECK-LABEL: func.func @MinimumSplitOverH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x2048x2048xf16>
func.func @MinimumSplitOverH(
    %arg0: tensor<1x32x2048x2048xf16>)
            -> tensor<1x32x2048x2048xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    %0 = VPU.Minimum(%arg0, %cst) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} :
        tensor<1x32x2048x2048xf16>, tensor<1x1x1x1xf16>
        -> tensor<1x32x2048x2048xf16>

    return %0 : tensor<1x32x2048x2048xf16>

    // CHECK:  [[CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf16>
    // CHECK:  [[Minimum0:%.+]] = VPU.Minimum(%arg0, [[CST]])
    // CHECK-SAME:  multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
    // CHECK-SAME:  tilingStrategy = [1, 1, 147, 1]
    // CHECH-SAME:  : tensor<1x32x2048x2048xf16>, tensor<1x1x1x1xf16>
    // CHECH-SAME:  -> tensor<1x32x2048x2048xf16>

    // CHECK:  return [[Minimum0]] : tensor<1x32x2048x2048xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @NoTilingClusterNCEConv
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
func.func @NoTilingClusterNCEConv(%arg0: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>) -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %weights = const.Declare tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}> = dense<1.000000e+00> : tensor<128x32x3x3xf16, {mem_space = @CMX_NN}>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}> = dense<10> : tensor<128x1x1x4xsi32, {mem_space = @CMX_NN}>

    %0 = VPU.NCE.ClusterTiling (
            %arg0 as %arg1: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            %weights as %arg2: tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>,
            %weights_table as %arg3: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
                -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}> {
      %1 = VPU.NCE.Convolution(%arg1, %arg2, %arg3) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                rawFilterShape = [128, 32, 3, 3],
                strides = [1, 1]
            } -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
      VPU.Yield %1
    }

    return %0 : tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK-DAG:        [[WEIGHTS:%.+]] = const.Declare tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-DAG:        [[WEIGHT_TABLE:%.+]] = const.Declare tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>

    // CHECK:        [[CLUSTER_TILING:%.+]] = VPU.NCE.ClusterTiling (
    // CHECK-SAME:          %arg0 as %arg1: tensor<1x32x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          [[WEIGHTS]] as %arg2: tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK-SAME:          [[WEIGHT_TABLE]] as %arg3: tensor<128x1x1x4xsi32, {mem_space = @CMX_NN, order = #NCHW}>)
    // CHECK-SAME:          -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           [[NCE_CONV:%.*]] = VPU.NCE.Convolution(%arg1, %arg2, %arg3)
    // CHECK-SAME:              pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-NOT:               tilingStrategy
    // CHECK-SAME:              -> tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:           VPU.Yield [[NCE_CONV]]

    // CHECK:         return [[CLUSTER_TILING]] : tensor<1x128x100x100xf16, {mem_space = @CMX_NN, order = #NHWC}>
}

// -----

// CHECK-LABEL: func.func @GatherSplit
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<4004x320xf16>
func.func @GatherSplit(%arg0: tensor<4004x320xf16>) -> tensor<4004x1xf16> {
    %cst = const.Declare tensor<1xsi32> = dense<4003> : tensor<1xsi64>, [#const.CastElemType<si32>]
    %0 = VPU.Gather(%arg0, %cst) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<4004x320xf16>, tensor<1xsi32> -> tensor<4004x1xf16>
    return %0 : tensor<4004x1xf16>

    // CHECK-DAG: [[VAL_1:%.+]] = const.Declare tensor<1xsi32> = dense<4003> : tensor<1xsi64>, [#const.CastElemType<si32>]

    // CHECK:     [[Gather0:%.+]] = VPU.Gather([[INPUT]], [[VAL_1]]) {axis_value = 1 : i64, batch_dims = 0 : i64, tilingStrategy = [2, 1]} : tensor<4004x320xf16>, tensor<1xsi32> -> tensor<4004x1xf16>

    // CHECK:     return [[Gather0]]
}

// -----

// CHECK-LABEL: func.func @GatherSplitWithBatchDims
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<2x4004x320xf16>
func.func @GatherSplitWithBatchDims(%arg0: tensor<2x4004x320xf16>) -> tensor<2x1x320xf16> {
    %cst = const.Declare tensor<2x1xsi32> = dense<1> : tensor<2x1xsi64>, [#const.CastElemType<si32>]
    %0 = VPU.Gather(%arg0, %cst) {axis_value = 1 : i64, batch_dims = 1 : i64} : tensor<2x4004x320xf16>, tensor<2x1xsi32> -> tensor<2x1x320xf16>
    return %0 : tensor<2x1x320xf16>

    // CHECK-DAG: [[VAL_1:%.+]] = const.Declare tensor<2x1xsi32> = dense<1> : tensor<2x1xsi64>, [#const.CastElemType<si32>]

    // CHECK:     [[Gather0:%.+]] = VPU.Gather([[INPUT]], [[VAL_1]]) {axis_value = 1 : i64, batch_dims = 1 : i64, tilingStrategy = [2, 1, 2]} : tensor<2x4004x320xf16>, tensor<2x1xsi32> -> tensor<2x1x320xf16>

    // CHECK:     return [[Gather0]]
}

// -----

// CHECK-LABEL: func.func @GatherSplitOptimize
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<387072x3xf16>
func.func @GatherSplitOptimize(%arg0: tensor<387072x3xf16>) -> tensor<1x387072x3xf16> {
    %cst = const.Declare tensor<1x387072xsi32> = dense<1> : tensor<1x387072xsi64>, [#const.CastElemType<si32>]
    %0 = VPU.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x3xf16>, tensor<1x387072xsi32> -> tensor<1x387072x3xf16>
    return %0 : tensor<1x387072x3xf16>

    // CHECK-DAG: [[VAL_1:%.+]] = const.Declare tensor<1x387072xsi32> = dense<1> : tensor<1x387072xsi64>, [#const.CastElemType<si32>]

    // CHECK:     [[Gather0:%.+]] = VPU.Gather([[INPUT]], [[VAL_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, tilingStrategy = [1, 2, 3]} : tensor<387072x3xf16>, tensor<1x387072xsi32> -> tensor<1x387072x3xf16>

    // CHECK:     return [[Gather0]]
}

// -----

// CHECK-LABEL: func.func @NotSplitGatherForLargeSizeOnGatherAxis
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1548288x3xf16>
func.func @NotSplitGatherForLargeSizeOnGatherAxis(%arg0: tensor<1548288x3xf16>) -> tensor<1x1548288x3xf16> {
    %cst = const.Declare tensor<1x1548288xsi32> = dense<1> : tensor<1x1548288xsi64>, [#const.CastElemType<si32>]
    %0 = VPU.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<1548288x3xf16>, tensor<1x1548288xsi32> -> tensor<1x1548288x3xf16>
    return %0 : tensor<1x1548288x3xf16>

    // CHECK-DAG: [[VAL_1:%.+]] = const.Declare tensor<1x1548288xsi32> = dense<1> : tensor<1x1548288xsi64>, [#const.CastElemType<si32>]

    // CHECK:     [[Gather0:%.+]] = VPU.Gather([[INPUT]], [[VAL_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<1548288x3xf16>, tensor<1x1548288xsi32> -> tensor<1x1548288x3xf16>

    // CHECK:     return [[Gather0]]
}

// -----

// CHECK-LABEL: func.func @NotSplitGatherForLargeIORatio
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<51865x512xf16>
func.func @NotSplitGatherForLargeIORatio(%arg0: tensor<51865x512xf16>) -> tensor<1x1x512xf16> {
    %cst = const.Declare tensor<1x1xsi32> = dense<1> : tensor<1x1xsi64>, [#const.CastElemType<si32>]
    %0 = VPU.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<51865x512xf16>, tensor<1x1xsi32> -> tensor<1x1x512xf16>
    return %0 : tensor<1x1x512xf16>

    // CHECK-DAG: [[VAL_1:%.+]] = const.Declare tensor<1x1xsi32> = dense<1> : tensor<1x1xsi64>, [#const.CastElemType<si32>]

    // CHECK:     [[Gather0:%.+]] = VPU.Gather([[INPUT]], [[VAL_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<51865x512xf16>, tensor<1x1xsi32> -> tensor<1x1x512xf16>

    // CHECK:     return [[Gather0]]
}

// -----

// CHECK-LABEL: func.func @Gather4DSplit
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x4004x320x1xf16>
func.func @Gather4DSplit(%arg0: tensor<1x4004x320x1xf16>) -> tensor<1x4004x1x1xf16> {
    %cst = const.Declare tensor<1x1x1x1xsi32> = dense<4003> : tensor<1x1x1x1xsi32>
    %0 = VPU.Gather(%arg0, %cst) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x4004x320x1xf16>, tensor<1x1x1x1xsi32> -> tensor<1x4004x1x1xf16>
    return %0 : tensor<1x4004x1x1xf16>

    // CHECK-DAG: [[INDICES:%.+]] = const.Declare tensor<1x1x1x1xsi32> = dense<4003> : tensor<1x1x1x1xsi32>
    // CHECK:     [[GATHER:%.+]] = VPU.Gather([[INPUT]], [[INDICES]]) {
    // CHECK-SAME:          axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64, tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      } : tensor<1x4004x320x1xf16>, tensor<1x1x1x1xsi32> -> tensor<1x4004x1x1xf16>

    // CHECK:     return [[GATHER]]
}

// -----

// CHECK-LABEL: func.func @Gather4DSplitWithBatchDims
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<2x1x4004x320xf16>
func.func @Gather4DSplitWithBatchDims(%arg0: tensor<2x1x4004x320xf16>) -> tensor<2x1x1x320xf16> {
    %cst = const.Declare tensor<2x1x1x1xsi32> = dense<1> : tensor<2x1x1x1xsi32>
    %0 = VPU.Gather(%arg0, %cst) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<2x1x4004x320xf16>, tensor<2x1x1x1xsi32> -> tensor<2x1x1x320xf16>
    return %0 : tensor<2x1x1x320xf16>

    // CHECK-DAG: [[INDICES:%.+]] = const.Declare tensor<2x1x1x1xsi32> = dense<1> : tensor<2x1x1x1xsi32>
    // CHECK:     [[GATHER:%.+]] = VPU.Gather([[INPUT]], [[INDICES]]) {
    // CHECK-SAME:          axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64, tilingStrategy = [2, 1, 1, 2]
    // CHECK-SAME:      } : tensor<2x1x4004x320xf16>, tensor<2x1x1x1xsi32> -> tensor<2x1x1x320xf16>

    // CHECK:     return [[GATHER]]
}

// -----

// CHECK-LABEL: func.func @Gather4DSplitOptimize
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x387072x3xf16>
func.func @Gather4DSplitOptimize(%arg0: tensor<1x1x387072x3xf16>) -> tensor<1x1x387072x3xf16> {
    %cst = const.Declare tensor<1x387072x1x1xsi32> = dense<1> : tensor<1x387072x1x1xsi32>
    %0 = VPU.Gather(%arg0, %cst) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x387072x3xf16>, tensor<1x387072x1x1xsi32> -> tensor<1x1x387072x3xf16>
    return %0 : tensor<1x1x387072x3xf16>

    // CHECK-DAG: [[INDICES:%.+]] = const.Declare tensor<1x387072x1x1xsi32> = dense<1> : tensor<1x387072x1x1xsi32>
    // CHECK:     [[GATHER:%.+]] = VPU.Gather([[INPUT]], [[INDICES]]) {
    // CHECK-SAME:          axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64, tilingStrategy = [1, 1, 2, 3]
    // CHECK-SAME:      } : tensor<1x1x387072x3xf16>, tensor<1x387072x1x1xsi32> -> tensor<1x1x387072x3xf16>

    // CHECK:     return [[GATHER]]
}

// -----

// CHECK-LABEL: func.func @NotSplitGather4DForLargeSizeOnGatherAxis
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x1548288x3xf16>
func.func @NotSplitGather4DForLargeSizeOnGatherAxis(%arg0: tensor<1x1x1548288x3xf16>) -> tensor<1x1x1548288x3xf16> {
    %cst = const.Declare tensor<1x1548288x1x1xsi32> = dense<1> : tensor<1x1548288x1x1xsi32>
    %0 = VPU.Gather(%arg0, %cst) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x1548288x3xf16>, tensor<1x1548288x1x1xsi32> -> tensor<1x1x1548288x3xf16>
    return %0 : tensor<1x1x1548288x3xf16>

    // CHECK-DAG: [[INDICES:%.+]] = const.Declare tensor<1x1548288x1x1xsi32> = dense<1> : tensor<1x1548288x1x1xsi32>
    // CHECK:     [[GATHER:%.+]] = VPU.Gather([[INPUT]], [[INDICES]]) {
    // CHECK-SAME:          axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64
    // CHECK-SAME:      } : tensor<1x1x1548288x3xf16>, tensor<1x1548288x1x1xsi32> -> tensor<1x1x1548288x3xf16>

    // CHECK:     return [[GATHER]]
}

// -----

// CHECK-LABEL: func.func @NotSplitGather4DForLargeIORatioUseDDRAccess
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x51865x512xf16>
func.func @NotSplitGather4DForLargeIORatioUseDDRAccess(%arg0: tensor<1x1x51865x512xf16>) -> tensor<1x1x1x512xf16> {
    %cst = const.Declare tensor<1x1x1x1xsi32> = dense<1> : tensor<1x1x1x1xsi32>
    %0 = VPU.Gather(%arg0, %cst) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x51865x512xf16>, tensor<1x1x1x1xsi32> -> tensor<1x1x1x512xf16>
    return %0 : tensor<1x1x1x512xf16>

    // CHECK-DAG: [[INDICES:%.+]] = const.Declare tensor<1x1x1x1xsi32> = dense<1> : tensor<1x1x1x1xsi32>
    // CHECK:     [[GATHER:%.+]] = VPU.Gather([[INPUT]], [[INDICES]]) {
    // CHECK-SAME:          axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64
    // CHECK-SAME:      } : tensor<1x1x51865x512xf16>, tensor<1x1x1x1xsi32> -> tensor<1x1x1x512xf16>

    // CHECK:     return [[GATHER]]
}

// -----

// CHECK-LABEL: func.func @Yuv2RGBSplit
func.func @Yuv2RGBSplit(%arg0: tensor<1x993x982x1xf16>) -> tensor<1x662x982x3xf16> {
    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 662, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x662x982x1xf16>
    %1 = VPU.Slice %arg0 [0, 662, 0, 0] [1, 331, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x331x982x1xf16>
    %2 = VPU.Reshape(%1) {shape_value = [1, 331, 491, 2]} : tensor<1x331x982x1xf16> -> tensor<1x331x491x2xf16>
    %3 = VPU.YuvToRgb(%0, %2) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x662x982x1xf16>, tensor<1x331x491x2xf16> -> tensor<1x662x982x3xf16>
    return %3 : tensor<1x662x982x3xf16>

    // CHECK:    [[SLICE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 662, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x662x982x1xf16>
    // CHECK:    [[SLICE1:%.+]] = VPU.Slice %arg0 [0, 662, 0, 0] [1, 331, 982, 1] : tensor<1x993x982x1xf16> to tensor<1x331x982x1xf16>
    // CHECK:    [[RESHAPE:%.+]] = VPU.Reshape([[SLICE1]]) {shape_value = [1, 331, 491, 2]} : tensor<1x331x982x1xf16> -> tensor<1x331x491x2xf16>
    // CHECK:    [[YUV2RGB:%.+]] = VPU.YuvToRgb([[SLICE0]], [[RESHAPE]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>, tilingStrategy = [1, 3, 1, 1]} : tensor<1x662x982x1xf16>, tensor<1x331x491x2xf16> -> tensor<1x662x982x3xf16>
    // CHECK:    return [[YUV2RGB]] : tensor<1x662x982x3xf16>
}

// -----

// CHECK-LABEL: func.func @GatherNDSplit
func.func @GatherNDSplit(%arg0: tensor<3x5x512x512xf16>) -> tensor<3x1x100x512xf16> {
    %cst = const.Declare tensor<3x1x100x2xsi32> = dense<1> : tensor<3x1x100x2xsi32>
    %0 = VPU.GatherND(%arg0, %cst) {batch_dims = 1 : i64} : tensor<3x5x512x512xf16>, tensor<3x1x100x2xsi32> -> tensor<3x1x100x512xf16>
    return %0 : tensor<3x1x100x512xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<3x1x100x2xsi32> = dense<1> : tensor<3x1x100x2xsi32>
    // CHECK:       [[GATHER:%.+]] = VPU.GatherND(%arg0, [[CST]]) {
    // CHECK-SAME:               batch_dims = 1 : i64,
    // CHECK-SAME:               tilingStrategy = [3, 1, 1, 2]}
    // CHECK-SAME:           : tensor<3x5x512x512xf16>, tensor<3x1x100x2xsi32> -> tensor<3x1x100x512xf16>

    // CHECK: return [[GATHER]] : tensor<3x1x100x512xf16>
}

// -----

// CHECK-LABEL: func.func @GatherNDSplitIndices
func.func @GatherNDSplitIndices(%arg0: tensor<64x2xf16>) -> tensor<400000x2xf16> {
    %cst = const.Declare tensor<400000x1xsi32> = dense<1> : tensor<400000x1xsi32>
    %0 = VPU.GatherND(%arg0, %cst) {batch_dims = 0 : i64} : tensor<64x2xf16>, tensor<400000x1xsi32> -> tensor<400000x2xf16>
    return %0 : tensor<400000x2xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<400000x1xsi32> = dense<1> : tensor<400000x1xsi32>
    // CHECK:       [[GATHER:%.+]] = VPU.GatherND(%arg0, [[CST]]) {
    // CHECK-SAME:               batch_dims = 0 : i64,
    // CHECK-SAME:               tilingStrategy = [2, 1]}
    // CHECK-SAME:           : tensor<64x2xf16>, tensor<400000x1xsi32> -> tensor<400000x2xf16>

    // CHECK: return [[GATHER]] : tensor<400000x2xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @DepthToSpaceBlocksFirstSplit
func.func @DepthToSpaceBlocksFirstSplit(%arg0: tensor<1x480x10x120xf32, {order = #NHWC}>) -> tensor<1x30x40x480xf32, {order = #NHWC}> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x480x10x120xf32, {order = #NHWC}> -> tensor<1x480x10x120xf16, {order = #NHWC}>
    %1 = VPU.DepthToSpace(%0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x480x10x120xf16, {order = #NHWC}> -> tensor<1x30x40x480xf16, {order = #NHWC}>
    %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x30x40x480xf16, {order = #NHWC}> -> tensor<1x30x40x480xf32, {order = #NHWC}>
    return %2 : tensor<1x30x40x480xf32, {order = #NHWC}>

    // CHECK:       [[CONVERT0:%.+]] = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 2, 1, 1]} : tensor<1x480x10x120xf32, {order = #NHWC}> -> tensor<1x480x10x120xf16, {order = #NHWC}>
    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace([[CONVERT0]]) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x480x10x120xf16, {order = #NHWC}> -> tensor<1x30x40x480xf16, {order = #NHWC}>
    // CHECK:       [[CONVERT1:%.+]] = VPU.Convert([[D2S]]) {dstElemType = f32, tilingStrategy = [1, 1, 1, 2]} : tensor<1x30x40x480xf16, {order = #NHWC}> -> tensor<1x30x40x480xf32, {order = #NHWC}>

    // CHECK:       return [[CONVERT1]] : tensor<1x30x40x480xf32, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @DepthToSpaceDepthFirstSplit(%arg0
func.func @DepthToSpaceDepthFirstSplit(%arg0: tensor<1x480x10x120xf32, {order = #NHWC}>) -> tensor<1x30x40x480xf32, {order = #NHWC}> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x480x10x120xf32, {order = #NHWC}> -> tensor<1x480x10x120xf16, {order = #NHWC}>
    %1 = VPU.DepthToSpace(%0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x480x10x120xf16, {order = #NHWC}> -> tensor<1x30x40x480xf16, {order = #NHWC}>
    %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x30x40x480xf16, {order = #NHWC}> -> tensor<1x30x40x480xf32, {order = #NHWC}>
    return %2 : tensor<1x30x40x480xf32, {order = #NHWC}>

    // CHECK:       [[CONVERT0:%.+]] = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 2, 1, 1]} : tensor<1x480x10x120xf32, {order = #NHWC}> -> tensor<1x480x10x120xf16, {order = #NHWC}>
    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace([[CONVERT0]]) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x480x10x120xf16, {order = #NHWC}> -> tensor<1x30x40x480xf16, {order = #NHWC}>
    // CHECK:       [[CONVERT1:%.+]] = VPU.Convert([[D2S]]) {dstElemType = f32, tilingStrategy = [1, 1, 1, 2]} : tensor<1x30x40x480xf16, {order = #NHWC}> -> tensor<1x30x40x480xf32, {order = #NHWC}>

    // CHECK:       return [[CONVERT1]] : tensor<1x30x40x480xf32, {order = #NHWC}>
}

// -----

// CHECK-LABEL:   func.func @SpaceToDepthBlockFirstSplit
func.func @SpaceToDepthBlockFirstSplit(%arg0: tensor<1x48x160x80xf32>) -> tensor<1x768x40x20xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    %1 = VPU.SpaceToDepthOp(%0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} : tensor<1x48x160x80xf16> -> tensor<1x768x40x20xf16>
    %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x768x40x20xf16> -> tensor<1x768x40x20xf32>
    return %2 : tensor<1x768x40x20xf32>

    // CHECK:       [[CONVERT0:%.+]] = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    // CHECK:       [[S2D:%.+]] = VPU.SpaceToDepthOp([[CONVERT0]]) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, tilingStrategy = [1, 1, 1, 2]} : tensor<1x48x160x80xf16> -> tensor<1x768x40x20xf16>
    // CHECK:       [[CONVERT1:%.+]] = VPU.Convert([[S2D]]) {dstElemType = f32, tilingStrategy = [1, 2, 1, 1]} : tensor<1x768x40x20xf16> -> tensor<1x768x40x20xf32>

    // CHECK:       return [[CONVERT1]] : tensor<1x768x40x20xf32>
}

// -----

// CHECK-LABEL: func.func @SpaceToDepthDepthFirstSplit
func.func @SpaceToDepthDepthFirstSplit(%arg0: tensor<1x48x160x80xf32>) -> tensor<1x768x40x20xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    %1 = VPU.SpaceToDepthOp(%0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>} : tensor<1x48x160x80xf16> -> tensor<1x768x40x20xf16>
    %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x768x40x20xf16> -> tensor<1x768x40x20xf32>
    return %2 : tensor<1x768x40x20xf32>

    // CHECK:       [[CONVERT0:%.+]] = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 2, 1]} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    // CHECK:       [[S2D:%.+]] = VPU.SpaceToDepthOp([[CONVERT0]]) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>, tilingStrategy = [1, 1, 1, 2]} : tensor<1x48x160x80xf16> -> tensor<1x768x40x20xf16>
    // CHECK:       [[CONVERT1:%.+]] = VPU.Convert([[S2D]]) {dstElemType = f32, tilingStrategy = [1, 2, 1, 1]} : tensor<1x768x40x20xf16> -> tensor<1x768x40x20xf32>

    // CHECK:       return [[CONVERT1]] : tensor<1x768x40x20xf32>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SplitNCEConvOverOH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<256x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [256, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:        [[FILTER:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

    // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<256x1x1x4xsi32>

    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]}
    // CHECK-SAME:          -> tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK:        return [[CONV]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

// CHECK-LABEL:   @SplitQuantNCEConvOverOC
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64x!qElemType, {order = #NHWC}>
func.func @SplitQuantNCEConvOverOC(%arg0: tensor<1x32x64x64x!qElemType, {order = #NHWC}>) -> tensor<1x512x64x64x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<512x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<512x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<512x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [512, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x512x64x64x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x512x64x64x!qElemType1, {order = #NHWC}>

    // CHECK-DAG:        [[WEIGHTS:%.+]] = const.Declare tensor<512x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<512x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]

    // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<512x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<512x1x1x4xsi32>

    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [512, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:          -> tensor<1x512x64x64x!qElemType1, {order = #NHWC}>

    // CHECK:        return [[CONV]] : tensor<1x512x64x64x!qElemType1, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEMaxPoolOverH
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x200x200xf16, {order = #NHWC}>)
func.func @SplitNCEMaxPoolOverH(%arg0: tensor<1x16x200x200xf16, {order = #NHWC}>) -> tensor<1x16x200x200xf16, {order = #NHWC}> {
    %0 = VPU.NCE.MaxPool(%arg0) {
        ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
    } -> tensor<1x16x200x200xf16, {order = #NHWC}>

    return %0 : tensor<1x16x200x200xf16, {order = #NHWC}>

    // CHECK:       [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[INPUT]]) {
    // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:      tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:      } -> tensor<1x16x200x200xf16, {order = #NHWC}>

    // CHECK:       return [[MAXPOOL]] : tensor<1x16x200x200xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @SplitNCEEltwiseAddOverC
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x1024x24x24xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x1024x24x24xf16, {order = #NHWC}>
func.func @SplitNCEEltwiseAddOverC(
        %arg0: tensor<1x1024x24x24xf16, {order = #NHWC}>,
        %arg1: tensor<1x1024x24x24xf16, {order = #NHWC}>)
            -> tensor<1x1024x24x24xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x1024x24x24xf16, {order = #NHWC}>

    return %0 : tensor<1x1024x24x24xf16, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT1]], [[INPUT2]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      -> tensor<1x1024x24x24xf16, {order = #NHWC}>

    // CHECK:       return [[ELTWISE]] : tensor<1x1024x24x24xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEEltwiseAddSameInput
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x2048x14x14xf16, {order = #NHWC}>
func.func @SplitNCEEltwiseAddSameInput(%arg0: tensor<1x2048x14x14xf16, {order = #NHWC}>) -> tensor<1x2048x14x14xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg0) {
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x2048x14x14xf16, {order = #NHWC}>

    return %0 : tensor<1x2048x14x14xf16, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT]], [[INPUT]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      } -> tensor<1x2048x14x14xf16, {order = #NHWC}>

    // CHECK:       return [[ELTWISE]] : tensor<1x2048x14x14xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvertU8F32SplitOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x2x80x4000xui8, {order = #NHWC}>
func.func @ConvertU8F32SplitOverW(%arg0: tensor<1x2x80x4000xui8, {order = #NHWC}>) -> tensor<1x2x80x4000xf32, {order = #NHWC}> {
    %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x2x80x4000xui8, {order = #NHWC}> -> tensor<1x2x80x4000xf32, {order = #NHWC}>
    return %0 : tensor<1x2x80x4000xf32, {order = #NHWC}>

    // CHECK:       [[CONVERT:%.+]] = VPU.Convert([[INPUT]]) {
    // CHECK-SAME:      dstElemType = f32
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 2]
    // CHECK-SAME:      }> -> tensor<1x2x80x4000xf32, {order = #NHWC}>

    // CHECK:       return [[CONVERT]] : tensor<1x2x80x4000xf32, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @SigmoidSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16>
func.func @SigmoidSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Sigmoid(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Sigmoid([[INPUT]]) {
    // CHECK-SAME:          tilingStrategy = [1, 1, 1, 2]}
    // CHECK-SAME       : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @TanhSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16>
func.func @TanhSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Tanh(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Tanh([[INPUT]]) {
    // CHECK-SAME:          tilingStrategy = [1, 1, 1, 2]}
    // CHECK-SAME:      : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}


// -----

// CHECK-LABEL: @ExpSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16>
func.func @ExpSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Exp(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>
    // CHECK:       [[OUTPUT:%.+]] = VPU.Exp([[INPUT]]) {
    // CHECK-SAME:          tilingStrategy = [1, 1, 1, 2]}
    // CHECK-SAME:      tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @SqrtSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16>
func.func @SqrtSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Sqrt(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Sqrt([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @EluSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16>
func.func @EluSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Elu(%arg0) {x = 1.000000e+00 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Elu([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2], x = 1.000000e+00 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @ClampSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16>
func.func @ClampSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Clamp(%arg0) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Clamp([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @ReLUSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16>
func.func @ReLUSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.ReLU(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.ReLU([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @HSwishSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x16x80x1280xf16>) -> tensor<1x16x80x1280xf16>
func.func @HSwishSplitOverW(%arg0: tensor<1x16x80x1280xf16>) -> tensor<1x16x80x1280xf16> {
    %0 = VPU.HSwish(%arg0) : tensor<1x16x80x1280xf16> -> tensor<1x16x80x1280xf16>
    return %0 : tensor<1x16x80x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.HSwish([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 4]} : tensor<1x16x80x1280xf16> -> tensor<1x16x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x80x1280xf16>
}

// -----

// CHECK-LABEL: @SplitDivideEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16>
func.func @SplitDivideEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
    %0 = VPU.Divide(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
    return %0 : tensor<1x10x256x256xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Divide([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MemPermuteSplitNCHWToNHWC2Part
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x546x40x40xf16>) -> tensor<1x40x40x546xf16>
func.func @MemPermuteSplitNCHWToNHWC2Part(%arg0: tensor<1x546x40x40xf16>) -> tensor<1x40x40x546xf16> {
    %0 = VPU.MemPermute(%arg0) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x546x40x40xf16> -> tensor<1x40x40x546xf16>
    return %0 : tensor<1x40x40x546xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.MemPermute([[INPUT]]) {
    // CHECK-SAME:  dst_order = #NCHW, mem_perm = #NHWC, tilingStrategy = [1, 1, 1, 2]
    // CHECK-SAME:  } : tensor<1x546x40x40xf16> -> tensor<1x40x40x546xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x40x40x546xf16>
}

// -----

// CHECK-LABEL: @AvgPoolSwSplit2Part
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x1800x16xf16>) -> tensor<1x32x1789x16xf16>
func.func @AvgPoolSwSplit2Part(%arg0: tensor<1x32x1800x16xf16>) -> tensor<1x32x1789x16xf16> {
    %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x32x1800x16xf16> -> tensor<1x32x1789x16xf16>
    return %0 : tensor<1x32x1789x16xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.AvgPool([[INPUT]]) {
    // CHECK-SAME:  exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1], tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:  } : tensor<1x32x1800x16xf16> -> tensor<1x32x1789x16xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x32x1789x16xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitSparseNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x80x80xf16, {order = #NHWC}>
func.func @SplitSparseNCEConvOverOH(%arg0: tensor<1x32x80x80xf16, {order = #NHWC}>) -> tensor<1x160x80x80xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00> : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<160x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<160x1x1x384xi1>, is_weights>
    %weights_table = const.Declare tensor<160x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<160x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights_sparse, %weights_table) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [160, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x160x80x80xf16, {order = #NHWC}>

    return %0 : tensor<1x160x80x80xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTS:%.+]] =  const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]

    // CHECK-DAG:        [[WEIGHTS_SM:%.+]] = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK:        [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights} -> !VPU.SparseTensor<
    // CHECK-SAME:       data=tensor<160x32x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:       sparsity_map=tensor<160x1x1x384xi1>, is_weights

    // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<160x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<160x1x1x4xsi32>

    // CHECK:        [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SPARSE]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [160, 32, 3, 3],
    // CHECK-SAME:          strides = [1, 1], tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:          -> tensor<1x160x80x80xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x160x80x80xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

// CHECK-LABEL:   @SplitSparseQuantNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x80x80x!qElemType, {order = #NHWC}>
func.func @SplitSparseQuantNCEConvOverOH(%arg0: tensor<1x32x80x80x!qElemType, {order = #NHWC}>) -> tensor<1x320x80x80x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<320x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<320x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<320x1x1x384xi1> = dense<1.000000e+00> : tensor<320x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<320x32x3x3x!qElemType2, {order = #NHWC}>, sparsity_map=tensor<320x1x1x384xi1>, is_weights>
    %weights_table = const.Declare tensor<320x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<320x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights_sparse, %weights_table) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [320, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x320x80x80x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x320x80x80x!qElemType1, {order = #NHWC}>

    // CHECK-DAG:        [[WEIGHTS:%.+]] = const.Declare tensor<320x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<320x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>, #const.Sparsify<false>]

    // CHECK-DAG:        [[WEIGHTS_SM:%.+]] = const.Declare tensor<320x1x1x384xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<320x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK:        [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights} -> !VPU.SparseTensor<
    // CHECK-SAME:       data=tensor<320x32x3x3x!qElemType2, {order = #NHWC}>,
    // CHECK-SAME:       sparsity_map=tensor<320x1x1x384xi1>, is_weights

    // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<320x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<320x1x1x4xsi32>

    // CHECK:        [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SPARSE]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [320, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:          -> tensor<1x320x80x80x!qElemType1, {order = #NHWC}>

    // CHECK:        return [[OUTPUT]] : tensor<1x320x80x80x!qElemType1, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEAveragePoolOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x7x23040xf16, {order = #NHWC}>
func.func @SplitNCEAveragePoolOverW(%arg0: tensor<1x16x7x23040xf16, {order = #NHWC}>) -> tensor<1x16x1x23040xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {kernel_size = [7, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x16x1x23040xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x23040xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {kernel_size = [7, 1]
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x16x1x23040xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x23040xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: @SplitAveragePoolOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1x7x368640xf16>
func.func @SplitAveragePoolOverW(%arg0: tensor<1x1x7x368640xf16>) -> tensor<1x1x1x368640xf16> {
    %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x7x368640xf16> -> tensor<1x1x1x368640xf16>

    return %0 : tensor<1x1x1x368640xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.AvgPool([[INPUT]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x1x1x368640xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x1x1x368640xf16>
}

// -----

// CHECK-LABEL:   func.func @SplitGRUSequenceForward
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<2x100000x10xf16>
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<2x1x1xf16>
func.func @SplitGRUSequenceForward(%arg0: tensor<2x100000x10xf16>, %arg1: tensor<2x1x1xf16>) -> (tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>) {
      %cst = const.Declare tensor<1x3x10xf16> = dense<1.000000e+00> : tensor<1x3x10xf16>
      %cst_0 = const.Declare tensor<1x3x1xf16> = dense<1.000000e+00> : tensor<1x3x1xf16>
      %cst_1 = const.Declare tensor<1x4xf16> = dense<1.000000e+00> : tensor<1x4xf16>
      %middle_hidden_state, %output_hidden_state = VPU.GRUSequence(%arg0, %arg1, %cst, %cst_0, %cst_1) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 1 : i64, seq_length = 100000 : i64, should_linear_before_reset} : tensor<2x100000x10xf16>, tensor<2x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>
      return %middle_hidden_state, %output_hidden_state : tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x3x10xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x3x10xf16>

    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1x3x1xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x3x1xf16>

    // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1x4xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x4xf16>

    // CHECK:       [[OUTPUTY:%.+]], [[OUTPUTHO:%.+]] = VPU.GRUSequence([[INPUT0]], [[INPUT1]], [[CST]], [[CST0]], [[CST1]])
    // CHECK-SAME:          tilingStrategy = [2, 1, 2, 1]
    // CHECK-SAME:     : tensor<2x100000x10xf16>, tensor<2x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>

    // CHECK:       return [[OUTPUTY]], [[OUTPUTHO]] :  tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>
}

// -----

// CHECK-LABEL:   func.func @SplitGRUSequenceReverse
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<2x100000x10xf16>
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<2x1x1xf16>
func.func @SplitGRUSequenceReverse(%arg0: tensor<2x100000x10xf16>, %arg1: tensor<2x1x1xf16>) -> (tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>) {
      %cst = const.Declare tensor<1x3x10xf16> = dense<1.000000e+00> : tensor<1x3x10xf16>
      %cst_0 = const.Declare tensor<1x3x1xf16> = dense<1.000000e+00> : tensor<1x3x1xf16>
      %cst_1 = const.Declare tensor<1x4xf16> = dense<1.000000e+00> : tensor<1x4xf16>
      %middle_hidden_state, %output_hidden_state = VPU.GRUSequence(%arg0, %arg1, %cst, %cst_0, %cst_1) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<REVERSE>, hidden_size = 1 : i64, seq_length = 100000 : i64, should_linear_before_reset} : tensor<2x100000x10xf16>, tensor<2x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>
      return %middle_hidden_state, %output_hidden_state : tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x3x10xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x3x10xf16>

    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1x3x1xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x3x1xf16>

    // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1x4xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x4xf16>

    // CHECK:       [[OUTPUTY:%.+]], [[OUTPUTHO:%.+]] = VPU.GRUSequence([[INPUT0]], [[INPUT1]], [[CST]], [[CST0]], [[CST1]])
    // CHECK-SAME:          tilingStrategy = [2, 1, 2, 1]
    // CHECK-SAME:     : tensor<2x100000x10xf16>, tensor<2x1x1xf16>, tensor<1x3x10xf16>, tensor<1x3x1xf16>, tensor<1x4xf16> -> tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>

    // CHECK:       return [[OUTPUTY]], [[OUTPUTHO]] :  tensor<2x1x100000x1xf16>, tensor<2x1x1xf16>
}

// -----

// CHECK-LABEL:   func.func @SplitGRUSequenceBidirectional
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<2x100000x10xf16>
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<2x2x1xf16>
func.func @SplitGRUSequenceBidirectional(%arg0: tensor<2x100000x10xf16>, %arg1: tensor<2x2x1xf16>) -> (tensor<2x2x100000x1xf16>, tensor<2x2x1xf16>) {
      %cst = const.Declare tensor<2x3x10xf16> = dense<1.000000e+00> : tensor<2x3x10xf16>
      %cst_0 = const.Declare tensor<2x3x1xf16> = dense<1.000000e+00> : tensor<2x3x1xf16>
      %cst_1 = const.Declare tensor<2x4xf16> = dense<1.000000e+00> : tensor<2x4xf16>
      %middle_hidden_state, %output_hidden_state = VPU.GRUSequence(%arg0, %arg1, %cst, %cst_0, %cst_1) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, hidden_size = 1 : i64, seq_length = 100000 : i64, should_linear_before_reset} : tensor<2x100000x10xf16>, tensor<2x2x1xf16>, tensor<2x3x10xf16>, tensor<2x3x1xf16>, tensor<2x4xf16> -> tensor<2x2x100000x1xf16>, tensor<2x2x1xf16>
      return %middle_hidden_state, %output_hidden_state : tensor<2x2x100000x1xf16>, tensor<2x2x1xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<2x3x10xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<2x3x10xf16>

    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<2x3x1xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<2x3x1xf16>

    // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<2x4xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<2x4xf16>

    // CHECK:       [[OUTPUTY:%.+]], [[OUTPUTHO:%.+]] = VPU.GRUSequence([[INPUT0]], [[INPUT1]], [[CST]], [[CST0]], [[CST1]])
    // CHECK-SAME:          tilingStrategy = [2, 2, 2, 1]
    // CHECK-SAME:     : tensor<2x100000x10xf16>, tensor<2x2x1xf16>, tensor<2x3x10xf16>, tensor<2x3x1xf16>, tensor<2x4xf16> -> tensor<2x2x100000x1xf16>, tensor<2x2x1xf16>

    // CHECK:       return [[OUTPUTY]], [[OUTPUTHO]] :  tensor<2x2x100000x1xf16>, tensor<2x2x1xf16>
}

// -----

// CHECK-LABEL:   func.func @SplitGRUSequenceLastPartForward
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<100x1x100x120xf16>
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<100x1x40xf16>
func.func @SplitGRUSequenceLastPartForward(%arg0: tensor<100x1x100x120xf16>, %arg1: tensor<100x1x40xf16>) -> (tensor<100x1x100x40xf16>, tensor<100x1x40xf16>) {
      %cst = const.Declare tensor<1x120x40xf16> = dense<1.000000e+00> : tensor<1x120x40xf16>
      %cst_0 = const.Declare tensor<1x160xf16> = dense<1.000000e+00> : tensor<1x160xf16>
      %middle_hidden_state, %output_hidden_state = VPU.GRUSequenceLastPart(%arg0, %arg1, %cst, %cst_0) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<FORWARD>, hidden_size = 40 : i64, seq_length = 100 : i64, should_linear_before_reset} : tensor<100x1x100x120xf16>, tensor<100x1x40xf16>, tensor<1x120x40xf16>, tensor<1x160xf16> -> tensor<100x1x100x40xf16>, tensor<100x1x40xf16>
      return %middle_hidden_state, %output_hidden_state : tensor<100x1x100x40xf16>, tensor<100x1x40xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x120x40xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x120x40xf16>

    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1x160xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x160xf16>

    // CHECK:       [[OUTPUTY:%.+]], [[OUTPUTHO:%.+]] = VPU.GRUSequenceLastPart([[INPUT0]], [[INPUT1]], [[CST]], [[CST0]])
    // CHECK-SAME:          tilingStrategy = [2, 1, 1, 1]
    // CHECK-SAME:     : tensor<100x1x100x120xf16>, tensor<100x1x40xf16>, tensor<1x120x40xf16>, tensor<1x160xf16> -> tensor<100x1x100x40xf16>, tensor<100x1x40xf16>

    // CHECK:       return [[OUTPUTY]], [[OUTPUTHO]] :  tensor<100x1x100x40xf16>, tensor<100x1x40xf16>
}

// -----

// CHECK-LABEL:   func.func @SplitGRUSequenceLastPartReverse
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<100x1x100x120xf16>
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<100x1x40xf16>
func.func @SplitGRUSequenceLastPartReverse(%arg0: tensor<100x1x100x120xf16>, %arg1: tensor<100x1x40xf16>) -> (tensor<100x1x100x40xf16>, tensor<100x1x40xf16>) {
      %cst = const.Declare tensor<1x120x40xf16> = dense<1.000000e+00> : tensor<1x120x40xf16>
      %cst_0 = const.Declare tensor<1x160xf16> = dense<1.000000e+00> : tensor<1x160xf16>
      %middle_hidden_state, %output_hidden_state = VPU.GRUSequenceLastPart(%arg0, %arg1, %cst, %cst_0) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<REVERSE>, hidden_size = 40 : i64, seq_length = 100 : i64, should_linear_before_reset} : tensor<100x1x100x120xf16>, tensor<100x1x40xf16>, tensor<1x120x40xf16>, tensor<1x160xf16> -> tensor<100x1x100x40xf16>, tensor<100x1x40xf16>
      return %middle_hidden_state, %output_hidden_state : tensor<100x1x100x40xf16>, tensor<100x1x40xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x120x40xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x120x40xf16>

    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<1x160xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<1x160xf16>

    // CHECK:       [[OUTPUTY:%.+]], [[OUTPUTHO:%.+]] = VPU.GRUSequenceLastPart([[INPUT0]], [[INPUT1]], [[CST]], [[CST0]])
    // CHECK-SAME:          tilingStrategy = [2, 1, 1, 1]
    // CHECK-SAME:     : tensor<100x1x100x120xf16>, tensor<100x1x40xf16>, tensor<1x120x40xf16>, tensor<1x160xf16> -> tensor<100x1x100x40xf16>, tensor<100x1x40xf16>

    // CHECK:       return [[OUTPUTY]], [[OUTPUTHO]] :  tensor<100x1x100x40xf16>, tensor<100x1x40xf16>
}

// -----

// CHECK-LABEL:   func.func @SplitGRUSequenceLastPartBidirectional
// CHECK-SAME:      [[INPUT0:%arg[0-9]]]: tensor<100x2x100x120xf16>
// CHECK-SAME:      [[INPUT1:%arg[0-9]]]: tensor<100x2x40xf16>
func.func @SplitGRUSequenceLastPartBidirectional(%arg0: tensor<100x2x100x120xf16>, %arg1: tensor<100x2x40xf16>) -> (tensor<100x2x100x40xf16>, tensor<100x2x40xf16>) {
      %cst = const.Declare tensor<2x120x40xf16> = dense<1.000000e+00> : tensor<2x120x40xf16>
      %cst_0 = const.Declare tensor<2x160xf16> = dense<1.000000e+00> : tensor<2x160xf16>
      %middle_hidden_state, %output_hidden_state = VPU.GRUSequenceLastPart(%arg0, %arg1, %cst, %cst_0) {clip = 0.000000e+00 : f64, direction = #IE.rnn_seq_direction<BIDIRECTIONAL>, hidden_size = 40 : i64, seq_length = 100 : i64, should_linear_before_reset} : tensor<100x2x100x120xf16>, tensor<100x2x40xf16>, tensor<2x120x40xf16>, tensor<2x160xf16> -> tensor<100x2x100x40xf16>, tensor<100x2x40xf16>
      return %middle_hidden_state, %output_hidden_state : tensor<100x2x100x40xf16>, tensor<100x2x40xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<2x120x40xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<2x120x40xf16>

    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<2x160xf16>
    // CHECK-SAME:      = dense<1.000000e+00> : tensor<2x160xf16>

    // CHECK:       [[OUTPUTY:%.+]], [[OUTPUTHO:%.+]] = VPU.GRUSequenceLastPart([[INPUT0]], [[INPUT1]], [[CST]], [[CST0]])
    // CHECK-SAME:          tilingStrategy = [4, 1, 1, 1]
    // CHECK-SAME:     : tensor<100x2x100x120xf16>, tensor<100x2x40xf16>, tensor<2x120x40xf16>, tensor<2x160xf16> -> tensor<100x2x100x40xf16>, tensor<100x2x40xf16>

    // CHECK:       return [[OUTPUTY]], [[OUTPUTHO]] :  tensor<100x2x100x40xf16>, tensor<100x2x40xf16>
}

// -----

// CHECK-LABEL:   func.func @GridSampleSplit
func.func @GridSampleSplit(%arg0: tensor<1x3x272x480xf16>, %arg1: tensor<1x272x480x2xf16>) -> tensor<1x3x272x480xf16> {
    %0 = VPU.GridSample(%arg0, %arg1) {align_corners, mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<BORDER>} : tensor<1x3x272x480xf16>, tensor<1x272x480x2xf16> -> tensor<1x3x272x480xf16>
    return %0 : tensor<1x3x272x480xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.GridSample(%arg0, %arg1)
    // CHECK-SAME:          tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:     : tensor<1x3x272x480xf16>, tensor<1x272x480x2xf16> -> tensor<1x3x272x480xf16>

    // CHECK:       return [[OUTPUT]] :  tensor<1x3x272x480xf16>
}

// -----

// CHECK-LABEL: func.func @SplitSwGroupConvOverOC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x256x96x96xf16>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<256x8x3x3xf16>
func.func @SplitSwGroupConvOverOC(
        %input: tensor<1x256x96x96xf16>,
        %filter: tensor<256x8x3x3xf16>)
            -> tensor<1x256x96x96xf16> {
    %1 = VPU.GroupConvolution(%input, %filter) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1],
        groups = 32 : i64

    } : tensor<1x256x96x96xf16>, tensor<256x8x3x3xf16> -> tensor<1x256x96x96xf16>
    return %1 : tensor<1x256x96x96xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.GroupConvolution([[INPUT]], [[FILTER]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 8, 1, 1]
    // CHECK-SAME:      -> tensor<1x256x96x96xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x96x96xf16>
}

// -----

// CHECK-LABEL: func.func @SplitSwGroupConvOverOCandOW
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x128x128x170xf16>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<128x128x9x9xf16>
func.func @SplitSwGroupConvOverOCandOW(
        %input: tensor<1x128x128x170xf16>,
        %filter: tensor<128x128x9x9xf16>)
            -> tensor<1x128x128x170xf16> {
    %1 = VPU.GroupConvolution(%input, %filter) {
        dilations = [1, 1],
        pads_begin = [4, 4],
        pads_end = [4, 4],
        strides = [1, 1],
        groups = 1 : i64

    } : tensor<1x128x128x170xf16>, tensor<128x128x9x9xf16> -> tensor<1x128x128x170xf16>
    return %1 : tensor<1x128x128x170xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.GroupConvolution([[INPUT]], [[FILTER]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [4, 4]
    // CHECK-SAME:          pads_end = [4, 4]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 2, 1, 170]
    // CHECK-SAME:      -> tensor<1x128x128x170xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x128x128x170xf16>
}

// -----

func.func @SplitSwGroupConvOverH(
        %input: tensor<1x32x256x256xf16>,
        %filter: tensor<64x1x3x3xf16>)
            -> tensor<1x64x256x256xf16> {
    %1 = VPU.GroupConvolution(%input, %filter) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1],
        groups = 32 : i64

    } : tensor<1x32x256x256xf16>, tensor<64x1x3x3xf16> -> tensor<1x64x256x256xf16>
    return %1 : tensor<1x64x256x256xf16>

    // CHECK-LABEL: func.func @SplitSwGroupConvOverH
    // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x256x256xf16>,
    // CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<64x1x3x3xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.GroupConvolution([[INPUT]], [[FILTER]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          groups = 32 : i64
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 1, 7, 1]
    // CHECK-SAME:      -> tensor<1x64x256x256xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x64x256x256xf16>
}

// -----

// CHECK-LABEL:   func.func @HardSigmoidSplit
func.func @HardSigmoidSplit(%arg0: tensor<1x16x192x192xf16>) -> tensor<1x16x192x192xf16> {
    %0 = VPU.HardSigmoid(%arg0) {alpha_value = 0.1666259765625 : f64, beta_value = 5.000000e-01 : f64} : tensor<1x16x192x192xf16> -> tensor<1x16x192x192xf16>
    return %0 : tensor<1x16x192x192xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.HardSigmoid(%arg0)
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:     :  tensor<1x16x192x192xf16> -> tensor<1x16x192x192xf16>

    // CHECK:       return [[OUTPUT]] :  tensor<1x16x192x192xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   func.func @MVN1NormalizeSplit
func.func @MVN1NormalizeSplit(%arg0: tensor<1x1x1x520001xf16>, %arg1: tensor<1x1x1x2xf16, {order = #NHWC}>) -> tensor<1x1x1x520001xf16> {
    %0 = VPU.MVN1Normalize(%arg0, %arg1) {across_channels = false, normalize_variance = true} : tensor<1x1x1x520001xf16>, tensor<1x1x1x2xf16, {order = #NHWC}> -> tensor<1x1x1x520001xf16>
    return %0 : tensor<1x1x1x520001xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.MVN1Normalize(%arg0, %arg1)
    // CHECK-SAME:          tilingStrategy = [1, 1, 1, 2]
    // CHECK-SAME:     :  tensor<1x1x1x520001xf16>, tensor<1x1x1x2xf16, {order = #NHWC}> -> tensor<1x1x1x520001xf16>

    // CHECK:       return [[OUTPUT]] :  tensor<1x1x1x520001xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   func.func @MVN1NormalizeSplitOverH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x512x256x256xf16, {order = #NHWC}>, [[MEAN_VAR:%.+]]: tensor<1x512x1x32xf16, {order = #NHWC}>
func.func @MVN1NormalizeSplitOverH(%arg0: tensor<1x512x256x256xf16, {order = #NHWC}>, %arg1: tensor<1x512x1x32xf16, {order = #NHWC}>) -> tensor<1x512x256x256xf16, {order = #NHWC}> {
    %0 = VPU.MVN1Normalize(%arg0, %arg1) {across_channels = false, normalize_variance = true} : tensor<1x512x256x256xf16, {order = #NHWC}>, tensor<1x512x1x32xf16, {order = #NHWC}> -> tensor<1x512x256x256xf16, {order = #NHWC}>
    return %0 :  tensor<1x512x256x256xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.MVN1Normalize([[INPUT]], [[MEAN_VAR]])
    // CHECK-SAME:          tilingStrategy = [1, 1, 128, 1]
    // CHECK:       return [[OUTPUT]] : tensor<1x512x256x256xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitSparseDepthConvWithBigC
func.func @SplitSparseDepthConvWithBigC(%arg0: tensor<1x4080x40x40xf16, {order = #NHWC}>) -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>> {
    %cst0 = const.Declare tensor<4080x1x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4080x1x4x4xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<4080x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<4080x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.DepthConvolution(%arg0, %cst0, %wt) {
            ppe = #VPU.PPEStub<>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [4080, 1, 4, 4],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>

    return %0 : !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>

    // CHECK-DAG: [[INPUT:%.+]] = const.Declare tensor<4080x1x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4080x1x4x4xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG: [[WT:%.*]] = const.Declare tensor<4080x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<4080x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    // CHECK: [[DWConv:%.+]] = VPU.NCE.DepthConvolution(%arg0, [[INPUT]], [[WT]]) {
    // CHECK:            tilingStrategy = [1, 14, 1, 1]
    // CHECK-SAME:     -> !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>
    // CHECK: return [[DWConv]] : !VPU.SparseTensor<data=tensor<1x4080x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4080x37x37xi1, {order = #NHWC}>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SplitSparseNCEMaxPoolWithBigC
func.func @SplitSparseNCEMaxPoolWithBigC(%arg0: tensor<1x4080x16x16xf16, {order = #NHWC}>) -> tensor<1x4080x16x16xf16, {order = #NHWC}> {
    %0 = VPU.Sparsify(%arg0) : tensor<1x4080x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>>
    %wt = const.Declare tensor<4080x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4080x1x1x4xsi32>
    %1 = VPU.NCE.MaxPool(%0, %wt) {
        ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
      } -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>>
    %2 = VPU.Desparsify(%1) : !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>> -> tensor<1x4080x16x16xf16, {order = #NHWC}>
    return %2 : tensor<1x4080x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0) : tensor<1x4080x16x16xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>>
    // CHECK-DAG: [[WT:%.+]] = const.Declare tensor<4080x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4080x1x1x4xsi32>
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.MaxPool([[VAL0]], [[WT]] )
    // CHECK:              tilingStrategy = [1, 3, 1, 1]
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x4080x16x16xf16, {order = #NHWC}>>
    // CHECK:       [[VAL2:%.+]] = VPU.Desparsify([[VAL1]])
    // CHECK:       return [[VAL2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitSparseDepthConvWithBigCWithSOK
func.func @SplitSparseDepthConvWithBigCWithSOK(%arg0: tensor<1x4112x40x40xf16, {order = #NHWC}>) -> !VPU.SparseTensor<data=tensor<1x4112x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4112x37x37xi1, {order = #NHWC}>> {
    %cst0 = const.Declare tensor<4112x1x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4112x1x4x4xf16>, [#const.Reorder<#NHWC>]
    %wt = const.Declare tensor<4112x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<4112x1x1x4xsi32>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.DepthConvolution(%arg0, %cst0, %wt) {
            ppe = #VPU.PPEStub<>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            rawFilterShape = [4112, 1, 4, 4],
            strides = [1, 1]
        } -> !VPU.SparseTensor<data=tensor<1x4112x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4112x37x37xi1, {order = #NHWC}>>

    return %0 : !VPU.SparseTensor<data=tensor<1x4112x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4112x37x37xi1, {order = #NHWC}>>

    // CHECK-DAG: [[INPUT:%.+]] = const.Declare tensor<4112x1x4x4xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<4112x1x4x4xf16>, [#const.Reorder<#NHWC>]
    // CHECK-DAG: [[WT:%.*]] = const.Declare tensor<4112x1x1x4xsi32, {order = #NHWC}> = dense<10> : tensor<4112x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    // CHECK: [[DWConv:%.+]] = VPU.NCE.DepthConvolution(%arg0, [[INPUT]], [[WT]]) {
    // CHECK:            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK:            tilingStrategy = [1, 7, 1, 1]
    // CHECK-SAME:     -> !VPU.SparseTensor<data=tensor<1x4112x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4112x37x37xi1, {order = #NHWC}>>
    // CHECK: return [[DWConv]] : !VPU.SparseTensor<data=tensor<1x4112x37x37xf16, {order = #NHWC}>, sparsity_map=tensor<1x4112x37x37xi1, {order = #NHWC}>>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SplitSparseNCEMaxPoolWithBigCWithSOK
func.func @SplitSparseNCEMaxPoolWithBigCWithSOK(%arg0: tensor<1x4112x16x16xf16, {order = #NHWC}>) -> tensor<1x4112x16x16xf16, {order = #NHWC}> {
    %0 = VPU.Sparsify(%arg0) : tensor<1x4112x16x16xf16, {order = #NHWC}> -> !VPU.SparseTensor<data=tensor<1x4112x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x4112x16x16xi1, {order = #NHWC}>>
    %wt = const.Declare tensor<4112x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4112x1x1x4xsi32>
    %1 = VPU.NCE.MaxPool(%0, %wt) {
        ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
      } -> !VPU.SparseTensor<data=tensor<1x4112x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x4112x16x16xi1, {order = #NHWC}>>
    %2 = VPU.Desparsify(%1) : !VPU.SparseTensor<data=tensor<1x4112x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x4112x16x16xi1, {order = #NHWC}>> -> tensor<1x4112x16x16xf16, {order = #NHWC}>
    return %2 : tensor<1x4112x16x16xf16, {order = #NHWC}>

    // CHECK:       [[VAL0:%.+]] = VPU.Sparsify(%arg0) : tensor<1x4112x16x16xf16, {order = #NHWC}>
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x4112x16x16xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x4112x16x16xi1, {order = #NHWC}>>
    // CHECK-DAG: [[WT:%.+]] = const.Declare tensor<4112x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4112x1x1x4xsi32>
    // CHECK:       [[VAL1:%.+]] = VPU.NCE.MaxPool([[VAL0]], [[WT]] )
    // CHECK:              multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK:              tilingStrategy = [1, 3, 1, 1]
    // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<1x4112x16x16xf16, {order = #NHWC}>,
    // CHECK-SAME:                           sparsity_map=tensor<1x4112x16x16xi1, {order = #NHWC}>>
    // CHECK:       [[VAL2:%.+]] = VPU.Desparsify([[VAL1]])
    // CHECK:       return [[VAL2]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!SparseType = !VPU.SparseTensor<data=tensor<1x2032x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x2032x16x16xi1, {order = #NHWC}>>
!SparseType1 = !VPU.SparseTensor<data=tensor<1x4064x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x4064x16x16xi1, {order = #NHWC}>>


// CHECK-LABEL: @SplitOutputSparseForConvSOKFollowedByConcat
func.func @SplitOutputSparseForConvSOKFollowedByConcat(%arg0: tensor<1x2032x16x16xf16, {order = #NHWC}>) -> tensor<1x4064x16x16xf16, {order = #NHWC}> {
    %s0 = VPU.Sparsify(%arg0) : tensor<1x2032x16x16xf16, {order = #NHWC}> -> !SparseType
    %wt0 = const.Declare tensor<2032x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<2032x1x1x4xsi32>
    %maxpool0 = VPU.NCE.MaxPool(%s0, %wt0) {
        ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
      } -> !SparseType

    %s1 = VPU.Sparsify(%arg0) : tensor<1x2032x16x16xf16, {order = #NHWC}> -> !SparseType
    %wt1 = const.Declare tensor<2032x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<2032x1x1x4xsi32>
    %maxpool1 = VPU.NCE.MaxPool(%s1, %wt1) {
        ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
      } -> !SparseType


    %concat = VPU.Concat(%maxpool0, %maxpool1) {static_offsets = [[0, 0, 0, 0], [0, 2032, 0, 0]]} : !SparseType, !SparseType -> !SparseType1

    %wt2 = const.Declare tensor<4064x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4064x1x1x4xsi32>
    %maxpool2 = VPU.NCE.MaxPool(%concat, %wt2) {
        ppe = #VPU.PPEStub<>,
        kernel_size = [3, 3],
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        strides = [1, 1]
      } -> !SparseType1

    %result = VPU.Desparsify(%maxpool2) : !SparseType1 -> tensor<1x4064x16x16xf16, {order = #NHWC}>
    return %result : tensor<1x4064x16x16xf16, {order = #NHWC}>

    // CHECK: [[ToSparsity_0:%.+]] = VPU.Sparsify(%arg0) : tensor<1x2032x16x16xf16, {order = #NHWC}>
    // CHECK:        -> !VPU.SparseTensor<data=tensor<1x2032x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x2032x16x16xi1, {order = #NHWC}>>
    // CHECK-DAG: [[WT_0:%.+]] = const.Declare tensor<2032x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<2032x1x1x4xsi32>
    // CHECK: [[MAXPOOL_0:%.+]] = VPU.NCE.MaxPool([[ToSparsity_0]], [[WT_0]] )
    // CHECK:              tilingStrategy = [1, 1, 2, 1]

    // CHECK: [[ToSparsity_1:%.+]] = VPU.Sparsify(%arg0) : tensor<1x2032x16x16xf16, {order = #NHWC}>
    // CHECK:        -> !VPU.SparseTensor<data=tensor<1x2032x16x16xf16, {order = #NHWC}>, sparsity_map=tensor<1x2032x16x16xi1, {order = #NHWC}>>
    // CHECK-DAG: [[WT_1:%.+]] = const.Declare tensor<2032x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<2032x1x1x4xsi32>
    // CHECK: [[MAXPOOL_1:%.+]] = VPU.NCE.MaxPool([[ToSparsity_1]], [[WT_1]] )
    // CHECK-SAME:              tilingStrategy = [1, 1, 2, 1]

    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[MAXPOOL_0]], [[MAXPOOL_1]])
    // CHECK-DAG: [[WT_2:%.+]] = const.Declare tensor<4064x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<4064x1x1x4xsi32>
    // CHECK: [[MAXPOOL_2:%.+]] = VPU.NCE.MaxPool([[CONCAT]], [[WT_2]] )
    // CHECK-SAME:              tilingStrategy = [1, 1, 3, 1]
    // CHECK: [[RESULT:%.+]] = VPU.Desparsify([[MAXPOOL_2]])

    // CHECK: return [[RESULT]]
}

// -----

// CHECK-LABEL: func.func @MVNTileOverCEvenly
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x163840x1xf16>

func.func @MVNTileOverCEvenly(%arg0: tensor<1x32x163840x1xf16>) -> tensor<1x32x163840x1xf16> {
    %0 = VPU.MVN(%arg0) {
        across_channels = false,
        eps = 9.9999997473787516E-6 : f64,
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        normalize_variance = true
    } : tensor<1x32x163840x1xf16> -> tensor<1x32x163840x1xf16>

    return %0 : tensor<1x32x163840x1xf16>

    // CHECK:    [[MVN:%.+]] = VPU.MVN([[INPUT]]) {
    // CHECK-SAME:          across_channels = false,
    // CHECK-SAME:          eps = 9.9999997473787516E-6 : f64,
    // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
    // CHECK-SAME:          normalize_variance = true,
    // CHECK-SAME:          tilingStrategy = [1, 8, 1, 1]
    // CHECK-NOT:           tilingStrategy = [1, 6, 1, 1]
    // CHECK-SAME:      } : tensor<1x32x163840x1xf16> -> tensor<1x32x163840x1xf16>

    // CHECK:    return [[MVN]] : tensor<1x32x163840x1xf16>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @AccumulateSplitOverH
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x4096x1024x1xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x4096x1024x1xf16, {order = #NHWC}>
func.func @AccumulateSplitOverH(%arg0: tensor<1x4096x1024x1xf16, {order = #NHWC}>, %arg1: tensor<1x4096x1024x1xf16, {order = #NHWC}>) -> tensor<1x4096x1024x1xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x4096x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x4096x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<1x4096x1x1xf16, {order = #NHWC}> = dense<2.000000e+00> : tensor<1x4096x1x1xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.Accumulate(%arg0, %arg1, %cst, %cst_0) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        } : tensor<1x4096x1024x1xf16, {order = #NHWC}>, tensor<1x4096x1024x1xf16, {order = #NHWC}>, tensor<1x4096x1x1xf16, {order = #NHWC}>, tensor<1x4096x1x1xf16, {order = #NHWC}> -> tensor<1x4096x1024x1xf16, {order = #NHWC}>

    return %0 : tensor<1x4096x1024x1xf16, {order = #NHWC}>

    // CHECK: [[LHS_SCALE:%.*]] = const.Declare tensor<1x4096x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x4096x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK: [[RHS_SCALE:%.*]] = const.Declare tensor<1x4096x1x1xf16, {order = #NHWC}> = dense<2.000000e+00> : tensor<1x4096x1x1xf16>, [#const.Reorder<#NHWC>]
    // CHECK: [[ACCUMULATE:%.+]] = VPU.Accumulate([[INPUT1]], [[INPUT2]], [[LHS_SCALE]], [[RHS_SCALE]]) {
    // CHECK-SAME:    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
    // CHECK-SAME:    tilingStrategy = [1, 11, 1, 1]
    // CHECK-SAME:    } : tensor<1x4096x1024x1xf16, {order = #NHWC}>, tensor<1x4096x1024x1xf16, {order = #NHWC}>, tensor<1x4096x1x1xf16, {order = #NHWC}>, tensor<1x4096x1x1xf16, {order = #NHWC}> -> tensor<1x4096x1024x1xf16, {order = #NHWC}>
    // CHECK: return [[ACCUMULATE]] : tensor<1x4096x1024x1xf16, {order = #NHWC}>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   func.func @MVN1MeanVarSplitTileAtC
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x256x8x512xf32, {order = #NHWC}>
func.func @MVN1MeanVarSplitTileAtC(%arg0: tensor<1x256x8x512xf32, {order = #NHWC}>) -> tensor<1x256x1x2xf16, {order = #NHWC}> {
    %0 = VPU.MVN1MeanVar(%arg0) {
        across_channels = false, eps = 9.9999999747524271E-7 : f64,
        normalize_variance = true,
        orig_shape = [1, 256, 1024, 1024],
        output_type = f16,
        multiClusterStrategy = #VPU.multi_cluster_strategy<Clustering>
    } : tensor<1x256x8x512xf32, {order = #NHWC}> -> tensor<1x256x1x2xf16, {order = #NHWC}>

    return %0 : tensor<1x256x1x2xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.MVN1MeanVar([[INPUT]])
    // CHECK-SAME:          tilingStrategy = [1, 3, 1, 1]
    // CHECK-SAME:     :  tensor<1x256x8x512xf32, {order = #NHWC}> -> tensor<1x256x1x2xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] :  tensor<1x256x1x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   func.func @MVN1MeanVarSplitTileAtN
// CHECK-SAME:      [[INPUT:%.+]]: tensor<256x1x8x512xf32, {order = #NHWC}>
func.func @MVN1MeanVarSplitTileAtN(%arg0: tensor<256x1x8x512xf32, {order = #NHWC}>) -> tensor<256x1x1x2xf16, {order = #NHWC}> {
    %0 = VPU.MVN1MeanVar(%arg0) {
        across_channels = true, eps = 9.9999999747524271E-7 : f64,
        normalize_variance = true,
        orig_shape = [256, 256, 1024, 1024],
        output_type = f16
    } : tensor<256x1x8x512xf32, {order = #NHWC}> -> tensor<256x1x1x2xf16, {order = #NHWC}>

    return %0 : tensor<256x1x1x2xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.MVN1MeanVar([[INPUT]])
    // CHECK-SAME:          tilingStrategy = [3, 1, 1, 1]
    // CHECK-SAME:     :  tensor<256x1x8x512xf32, {order = #NHWC}> -> tensor<256x1x1x2xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] :  tensor<256x1x1x2xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL:   func.func @NotSplitGridSample
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x1x1920x1088xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x1920x1088x2xf16>
func.func @NotSplitGridSample(%arg0: tensor<1x1x1920x1088xf16>, %arg1: tensor<1x1920x1088x2xf16>) -> tensor<1x1x1920x1088xf16> {
    %0 = VPU.GridSample(%arg0, %arg1) {align_corners, mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<BORDER>} : tensor<1x1x1920x1088xf16>, tensor<1x1920x1088x2xf16> -> tensor<1x1x1920x1088xf16>
    return %0 : tensor<1x1x1920x1088xf16>

    // CHECK:       [[GRID_SAMPLE:%.+]] = VPU.GridSample([[INPUT1]], [[INPUT2]]) {align_corners, mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<BORDER>}
    // CHECK-SAME:     : tensor<1x1x1920x1088xf16>, tensor<1x1920x1088x2xf16> -> tensor<1x1x1920x1088xf16>
    // CHECK:       return [[GRID_SAMPLE]] : tensor<1x1x1920x1088xf16>
}

// -----

// CHECK-LABEL:   func.func @SplitGridSample
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x4x960x544xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x960x544x2xf16>
func.func @SplitGridSample(%arg0: tensor<1x4x960x544xf16>, %arg1: tensor<1x960x544x2xf16>) -> tensor<1x4x960x544xf16> {
    %0 = VPU.GridSample(%arg0, %arg1) {align_corners, mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<BORDER>} : tensor<1x4x960x544xf16>, tensor<1x960x544x2xf16> -> tensor<1x4x960x544xf16>
    return %0 : tensor<1x4x960x544xf16>

    // CHECK:       [[GRID_SAMPLE:%.+]] = VPU.GridSample([[INPUT1]], [[INPUT2]]) {align_corners, mode = #IE.grid_sample_mode<BILINEAR>, padding_mode = #IE.grid_sample_padding_mode<BORDER>,
    // CHECK-SAME:          tilingStrategy = [1, 4, 4, 1]}
    // CHECK-SAME:     : tensor<1x4x960x544xf16>, tensor<1x960x544x2xf16> -> tensor<1x4x960x544xf16>

    // CHECK:       return [[GRID_SAMPLE]] : tensor<1x4x960x544xf16>
}
