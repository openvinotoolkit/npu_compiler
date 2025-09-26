//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --tiling-strategy-assignment="tiling-mode=ISOLATED" %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

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

    // CHECK:       [[OUTPUT:%.+]] = VPU.Convolution([[INPUT]], [[FILTER]], [[BIAS]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      -> tensor<1x256x64x64xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x64x64xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

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

    // CHECK:       [[OUTPUT:%.+]] = VPU.MaxPool([[INPUT]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:      -> tensor<1x16x200x200xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x200x200xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @SplitSwAddOverC
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x2048x14x14xf16>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x2048x14x14xf16>
func.func @SplitSwAddOverC(
        %input1: tensor<1x2048x14x14xf16>,
        %input2: tensor<1x2048x14x14xf16>)
            -> tensor<1x2048x14x14xf16> {
    %1 = VPU.Add(%input1, %input2) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x2048x14x14xf16>, tensor<1x2048x14x14xf16> -> tensor<1x2048x14x14xf16>
    return %1 : tensor<1x2048x14x14xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Add([[INPUT1]], [[INPUT2]])
    // CHECK-SAME:          tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      -> tensor<1x2048x14x14xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x2048x14x14xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @SplitAddSameInputOverC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x2048x14x14xf16>
func.func @SplitAddSameInputOverC(
        %input: tensor<1x2048x14x14xf16>)
            -> tensor<1x2048x14x14xf16> {
    %1 = VPU.Add(%input, %input) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } : tensor<1x2048x14x14xf16>, tensor<1x2048x14x14xf16> -> tensor<1x2048x14x14xf16>
    return %1 : tensor<1x2048x14x14xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Add([[INPUT]], [[INPUT]])
    // CHECK-SAME:          tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      -> tensor<1x2048x14x14xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x2048x14x14xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @InterpSplitOverC
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x24x64x64xf16>
func.func @InterpSplitOverC(
        %input1: tensor<1x24x64x64xf16>)
            -> tensor<1x24x256x256xf16> {

    %0 = VPU.Interpolate(%input1) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <LINEAR>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
            axes_attr = [2, 3], sizes_attr = [256, 256], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0> } :
        tensor<1x24x64x64xf16> -> tensor<1x24x256x256xf16>

    return %0 : tensor<1x24x256x256xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Interpolate([[INPUT1]]
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
    // CHECK-SAME:      tilingStrategy = [1, 1, 3, 1]
    // CHECK-SAME:      : tensor<1x24x64x64xf16>
    // CHECK-SAME:      -> tensor<1x24x256x256xf16>
    // CHECK:       return [[OUTPUT]] : tensor<1x24x256x256xf16>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @InterpSplitOverCNoCommonFactor
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x64x31x31xf16, {order = #NHWC}>
func.func @InterpSplitOverCNoCommonFactor(
    %arg0: tensor<1x64x31x31xf16, {order = #NHWC}>)
            -> tensor<1x64x121x121xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        sizes_attr = [121, 121],
        tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} :
        tensor<1x64x31x31xf16, {order = #NHWC}> -> tensor<1x64x121x121xf16, {order = #NHWC}>
    return %0 : tensor<1x64x121x121xf16, {order = #NHWC}>

    // CHECK:  [[INTERP0:%.+]] = VPU.Interpolate(%arg0)
    // CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:  : tensor<1x64x31x31xf16, {order = #NHWC}>
    // CHECK-SAME:  -> tensor<1x64x121x121xf16, {order = #NHWC}>

    // CHECK:  return [[INTERP0]] : tensor<1x64x121x121xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @InterpSplitOverHW
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x128x35x35xf16, {order = #NHWC}>
func.func @InterpSplitOverHW(
    %input1: tensor<1x128x35x35xf16, {order = #NHWC}>)
            -> tensor<1x128x168x335xf16, {order = #NHWC}> {
    %0 = VPU.Interpolate(%input1) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <LINEAR>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
        axes_attr = [2, 3],
        operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
        sizes_attr = [168, 335]} :
        tensor<1x128x35x35xf16, {order = #NHWC}> -> tensor<1x128x168x335xf16, {order = #NHWC}>
    return %0 : tensor<1x128x168x335xf16, {order = #NHWC}>

    // CHECK:  [[INTERP0:%.+]] = VPU.Interpolate(%arg0)
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 11]
    // CHECK-SAME:  : tensor<1x128x35x35xf16, {order = #NHWC}>
    // CHECK-SAME:  -> tensor<1x128x168x335xf16, {order = #NHWC}>

    // CHECK:  return [[INTERP0]] : tensor<1x128x168x335xf16, {order = #NHWC}>

}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!DistributedTensor0 = !VPU.DistributedTensor<
    1x32x100x100xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

!DistributedTensor1 = !VPU.DistributedTensor<
    1x128x100x100xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64
}>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL:   @NoTilingClusterNCEConv
// CHECK-SAME:          [[INPUT:%.+]]: !VPU.DistributedTensor<1x32x100x100xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
func.func @NoTilingClusterNCEConv(%arg0: !DistributedTensor0) -> !DistributedTensor1 {
    %weights = const.Declare tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}> = dense<1.000000e+00> : tensor<128x32x3x3xf16, {mem_space = @CMX_NN}>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %weights) {
                ppe = #VPU.PPEStub<>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                rawFilterShape = [128, 32, 3, 3],
                strides = [1, 1]
            } : !VPU.DistributedTensor<1x32x100x100xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>, tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}> -> !VPU.DistributedTensor<1x128x100x100xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    return %0 : !VPU.DistributedTensor<1x128x100x100xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK-DAG:        [[WEIGHTS:%.+]] = const.Declare tensor<128x32x3x3xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK:           [[NCE_CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]])
    // CHECK-SAME:              pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
    // CHECK-SAME:              strides = [1, 1]
    // CHECK-NOT:               tilingStrategy
    // CHECK-SAME:              -> !VPU.DistributedTensor<1x128x100x100xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:         return [[NCE_CONV]] : !VPU.DistributedTensor<1x128x100x100xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

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

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

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

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @GatherSplitOptimize
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<387072x3xf16>
func.func @GatherSplitOptimize(%arg0: tensor<387072x3xf16>) -> tensor<1x387072x3xf16> {
    %cst = const.Declare tensor<1x387072xsi32> = dense<1> : tensor<1x387072xsi64>, [#const.CastElemType<si32>]
    %0 = VPU.Gather(%arg0, %cst) {axis_value = 0 : i64, batch_dims = 0 : i64} : tensor<387072x3xf16>, tensor<1x387072xsi32> -> tensor<1x387072x3xf16>
    return %0 : tensor<1x387072x3xf16>

    // CHECK-DAG: [[VAL_1:%.+]] = const.Declare tensor<1x387072xsi32> = dense<1> : tensor<1x387072xsi64>, [#const.CastElemType<si32>]

    // CHECK:     [[Gather0:%.+]] = VPU.Gather([[INPUT]], [[VAL_1]]) {axis_value = 0 : i64, batch_dims = 0 : i64, tilingStrategy = [1, 4, 3]} : tensor<387072x3xf16>, tensor<1x387072xsi32> -> tensor<1x387072x3xf16>

    // CHECK:     return [[Gather0]]
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @Yuv2RGBSplit
func.func @Yuv2RGBSplit(%arg0: tensor<1x993x736x1xf16>) -> tensor<1x662x736x3xf16> {
    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 662, 736, 1] : tensor<1x993x736x1xf16> to tensor<1x662x736x1xf16>
    %1 = VPU.Slice %arg0 [0, 662, 0, 0] [1, 331, 736, 1] : tensor<1x993x736x1xf16> to tensor<1x331x736x1xf16>
    %2 = VPU.Reshape(%1) {shape_value = [1, 331, 368, 2]} : tensor<1x331x736x1xf16> -> tensor<1x331x368x2xf16>
    %3 = VPU.YuvToRgb(%0, %2) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>} : tensor<1x662x736x1xf16>, tensor<1x331x368x2xf16> -> tensor<1x662x736x3xf16>
    return %3 : tensor<1x662x736x3xf16>

    // CHECK:    [[SLICE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 662, 736, 1] : tensor<1x993x736x1xf16> to tensor<1x662x736x1xf16>
    // CHECK:    [[SLICE1:%.+]] = VPU.Slice %arg0 [0, 662, 0, 0] [1, 331, 736, 1] : tensor<1x993x736x1xf16> to tensor<1x331x736x1xf16>
    // CHECK:    [[RESHAPE:%.+]] = VPU.Reshape([[SLICE1]]) {shape_value = [1, 331, 368, 2]} : tensor<1x331x736x1xf16> -> tensor<1x331x368x2xf16>
    // CHECK:    [[YUV2RGB:%.+]] = VPU.YuvToRgb([[SLICE0]], [[RESHAPE]]) {inFmt = #IE.color_fmt<NV12>, operandSegmentSizes = array<i32: 1, 1, 0>, outFmt = #IE.color_fmt<RGB>, tilingStrategy = [1, 3, 1, 1]} : tensor<1x662x736x1xf16>, tensor<1x331x368x2xf16> -> tensor<1x662x736x3xf16>
    // CHECK:    return [[YUV2RGB]] : tensor<1x662x736x3xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

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

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @GatherNDSplitIndices
func.func @GatherNDSplitIndices(%arg0: tensor<64x2xf16>) -> tensor<300000x2xf16> {
    %cst = const.Declare tensor<300000x1xsi32> = dense<1> : tensor<300000x1xsi32>
    %0 = VPU.GatherND(%arg0, %cst) {batch_dims = 0 : i64} : tensor<64x2xf16>, tensor<300000x1xsi32> -> tensor<300000x2xf16>
    return %0 : tensor<300000x2xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<300000x1xsi32> = dense<1> : tensor<300000x1xsi32>
    // CHECK:       [[GATHER:%.+]] = VPU.GatherND(%arg0, [[CST]]) {
    // CHECK-SAME:               batch_dims = 0 : i64,
    // CHECK-SAME:               tilingStrategy = [2, 1]}
    // CHECK-SAME:           : tensor<64x2xf16>, tensor<300000x1xsi32> -> tensor<300000x2xf16>

    // CHECK: return [[GATHER]] : tensor<300000x2xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

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

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

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

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

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

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

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

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @Gather4DSplitOptimize
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x1x387072x3xf16>
func.func @Gather4DSplitOptimize(%arg0: tensor<1x1x387072x3xf16>) -> tensor<1x1x387072x3xf16> {
    %cst = const.Declare tensor<1x387072x1x1xsi32> = dense<1> : tensor<1x387072x1x1xsi32>
    %0 = VPU.Gather(%arg0, %cst) {
            axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64} : tensor<1x1x387072x3xf16>, tensor<1x387072x1x1xsi32> -> tensor<1x1x387072x3xf16>
    return %0 : tensor<1x1x387072x3xf16>

    // CHECK-DAG: [[INDICES:%.+]] = const.Declare tensor<1x387072x1x1xsi32> = dense<1> : tensor<1x387072x1x1xsi32>
    // CHECK:     [[GATHER:%.+]] = VPU.Gather([[INPUT]], [[INDICES]]) {
    // CHECK-SAME:          axis_value = 2 : i64, batch_dims = 1 : i64, indices_rank = 2 : i64, tilingStrategy = [1, 1, 4, 3]
    // CHECK-SAME:      } : tensor<1x1x387072x3xf16>, tensor<1x387072x1x1xsi32> -> tensor<1x1x387072x3xf16>

    // CHECK:     return [[GATHER]]
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

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

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

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

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @DepthToSpaceBlocksFirstSplit
func.func @DepthToSpaceBlocksFirstSplit(%arg0: tensor<1x384x10x120xf32, {order = #NHWC}>) -> tensor<1x24x40x480xf32, {order = #NHWC}> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x384x10x120xf32, {order = #NHWC}> -> tensor<1x384x10x120xf16, {order = #NHWC}>
    %1 = VPU.DepthToSpace(%0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>} : tensor<1x384x10x120xf16, {order = #NHWC}> -> tensor<1x24x40x480xf16, {order = #NHWC}>
    %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x24x40x480xf16, {order = #NHWC}> -> tensor<1x24x40x480xf32, {order = #NHWC}>
    return %2 : tensor<1x24x40x480xf32, {order = #NHWC}>

    // CHECK:       [[CONVERT0:%.+]] = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 2, 1, 1]} : tensor<1x384x10x120xf32, {order = #NHWC}> -> tensor<1x384x10x120xf16, {order = #NHWC}>
    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace([[CONVERT0]]) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x384x10x120xf16, {order = #NHWC}> -> tensor<1x24x40x480xf16, {order = #NHWC}>
    // CHECK:       [[CONVERT1:%.+]] = VPU.Convert([[D2S]]) {dstElemType = f32, tilingStrategy = [1, 1, 1, 2]} : tensor<1x24x40x480xf16, {order = #NHWC}> -> tensor<1x24x40x480xf32, {order = #NHWC}>

    // CHECK:       return [[CONVERT1]] : tensor<1x24x40x480xf32, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @DepthToSpaceDepthFirstSplit
func.func @DepthToSpaceDepthFirstSplit(%arg0: tensor<1x384x10x120xf32, {order = #NHWC}>) -> tensor<1x24x40x480xf32, {order = #NHWC}> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x384x10x120xf32, {order = #NHWC}> -> tensor<1x384x10x120xf16, {order = #NHWC}>
    %1 = VPU.DepthToSpace(%0) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>} : tensor<1x384x10x120xf16, {order = #NHWC}> -> tensor<1x24x40x480xf16, {order = #NHWC}>
    %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x24x40x480xf16, {order = #NHWC}> -> tensor<1x24x40x480xf32, {order = #NHWC}>
    return %2 : tensor<1x24x40x480xf32, {order = #NHWC}>

    // CHECK:       [[CONVERT0:%.+]] = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 2, 1, 1]} : tensor<1x384x10x120xf32, {order = #NHWC}> -> tensor<1x384x10x120xf16, {order = #NHWC}>
    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace([[CONVERT0]]) {block_size = 4 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x384x10x120xf16, {order = #NHWC}> -> tensor<1x24x40x480xf16, {order = #NHWC}>
    // CHECK:       [[CONVERT1:%.+]] = VPU.Convert([[D2S]]) {dstElemType = f32, tilingStrategy = [1, 1, 1, 2]} : tensor<1x24x40x480xf16, {order = #NHWC}> -> tensor<1x24x40x480xf32, {order = #NHWC}>

    // CHECK:       return [[CONVERT1]] : tensor<1x24x40x480xf32, {order = #NHWC}>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL:   func.func @SpaceToDepthBlockFirstSplit
func.func @SpaceToDepthBlockFirstSplit(%arg0: tensor<1x48x160x80xf16>) -> tensor<1x768x40x20xf16> {
    %0 = VPU.SpaceToDepthOp(%arg0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>} : tensor<1x48x160x80xf16> -> tensor<1x768x40x20xf16>
    return %0 : tensor<1x768x40x20xf16>

    // CHECK:       [[S2D:%.+]] = VPU.SpaceToDepthOp(%arg0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<BLOCKS_FIRST>, tilingStrategy = [1, 1, 1, 2]} : tensor<1x48x160x80xf16> -> tensor<1x768x40x20xf16>
    // CHECK:       return [[S2D]] : tensor<1x768x40x20xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @SpaceToDepthDepthFirstSplit
func.func @SpaceToDepthDepthFirstSplit(%arg0: tensor<1x48x160x80xf32>) -> tensor<1x768x40x20xf32> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    %1 = VPU.SpaceToDepthOp(%0) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>} : tensor<1x48x160x80xf16> -> tensor<1x768x40x20xf16>
    %2 = VPU.Convert(%1) {dstElemType = f32} : tensor<1x768x40x20xf16> -> tensor<1x768x40x20xf32>
    return %2 : tensor<1x768x40x20xf32>

    // CHECK:       [[CONVERT0:%.+]] = VPU.Convert(%arg0) {dstElemType = f16, tilingStrategy = [1, 1, 3, 1]} : tensor<1x48x160x80xf32> -> tensor<1x48x160x80xf16>
    // CHECK:       [[S2D:%.+]] = VPU.SpaceToDepthOp([[CONVERT0]]) {block_size = 4 : i64, mode = #IE.space_to_depth_mode<DEPTH_FIRST>, tilingStrategy = [1, 1, 1, 2]} : tensor<1x48x160x80xf16> -> tensor<1x768x40x20xf16>
    // CHECK:       [[CONVERT1:%.+]] = VPU.Convert([[S2D]]) {dstElemType = f32, tilingStrategy = [1, 3, 1, 1]} : tensor<1x768x40x20xf16> -> tensor<1x768x40x20xf32>

    // CHECK:       return [[CONVERT1]] : tensor<1x768x40x20xf32>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL:   @SplitNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x48xf16, {order = #NHWC}>
func.func @SplitNCEConvOverOH(%arg0: tensor<1x32x64x48xf16, {order = #NHWC}>) -> tensor<1x256x64x48xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %weights) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [256, 32, 3, 3],
        strides = [1, 1]
    } : tensor<1x32x64x48xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x64x48xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x48xf16, {order = #NHWC}>

    // CHECK-DAG:        [[FILTER:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]


    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:          -> tensor<1x256x64x48xf16, {order = #NHWC}>

    // CHECK:        return [[CONV]] : tensor<1x256x64x48xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL:   @SplitQuantNCEConvOverOC
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64x!qElemType, {order = #NHWC}>
func.func @SplitQuantNCEConvOverOC(%arg0: tensor<1x32x64x64x!qElemType, {order = #NHWC}>) -> tensor<1x512x64x64x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<512x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<512x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]

    %0 = VPU.NCE.Convolution(%arg0, %weights) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [512, 32, 3, 3],
        strides = [1, 1]
    } : tensor<1x32x64x64x!qElemType, {order = #NHWC}>, tensor<512x32x3x3x!qElemType2, {order = #NHWC}> -> tensor<1x512x64x64x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x512x64x64x!qElemType1, {order = #NHWC}>

    // CHECK-DAG:        [[WEIGHTS:%.+]] = const.Declare tensor<512x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<512x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>]


    // CHECK:        [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [512, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:          -> tensor<1x512x64x64x!qElemType1, {order = #NHWC}>

    // CHECK:        return [[CONV]] : tensor<1x512x64x64x!qElemType1, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

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

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @SplitNCEEltwiseAddOverC
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x1024x24x16xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT2:%arg[0-9]]]: tensor<1x1024x24x16xf16, {order = #NHWC}>
func.func @SplitNCEEltwiseAddOverC(
        %arg0: tensor<1x1024x24x16xf16, {order = #NHWC}>,
        %arg1: tensor<1x1024x24x16xf16, {order = #NHWC}>)
            -> tensor<1x1024x24x16xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        ppe = #VPU.PPEStub<>,
        op_type = #VPU.eltwise_type<ADD>
    } -> tensor<1x1024x24x16xf16, {order = #NHWC}>

    return %0 : tensor<1x1024x24x16xf16, {order = #NHWC}>

    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[INPUT1]], [[INPUT2]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      -> tensor<1x1024x24x16xf16, {order = #NHWC}>

    // CHECK:       return [[ELTWISE]] : tensor<1x1024x24x16xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

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

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @ConvertU8F32SplitOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x2x80x3000xui8, {order = #NHWC}>
func.func @ConvertU8F32SplitOverW(%arg0: tensor<1x2x80x3000xui8, {order = #NHWC}>) -> tensor<1x2x80x3000xf32, {order = #NHWC}> {
    %0 = VPU.Convert(%arg0) {dstElemType = f32} : tensor<1x2x80x3000xui8, {order = #NHWC}> -> tensor<1x2x80x3000xf32, {order = #NHWC}>
    return %0 : tensor<1x2x80x3000xf32, {order = #NHWC}>

    // CHECK:       [[CONVERT:%.+]] = VPU.Convert([[INPUT]]) {
    // CHECK-SAME:      dstElemType = f32
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 2]
    // CHECK-SAME:      }> -> tensor<1x2x80x3000xf32, {order = #NHWC}>

    // CHECK:       return [[CONVERT]] : tensor<1x2x80x3000xf32, {order = #NHWC}>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @SigmoidSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
func.func @SigmoidSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    %0 = VPU.Sigmoid(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
    return %0 : tensor<1x8x80x960xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Sigmoid([[INPUT]]) {
    // CHECK-SAME:          tilingStrategy = [1, 1, 1, 2]}
    // CHECK-SAME:      : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @TanhSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
func.func @TanhSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    %0 = VPU.Tanh(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
    return %0 : tensor<1x8x80x960xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Tanh([[INPUT]]) {
    // CHECK-SAME:          tilingStrategy = [1, 1, 1, 2]}
    // CHECK-SAME:      : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @ExpSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
func.func @ExpSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    %0 = VPU.Exp(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
    return %0 : tensor<1x8x80x960xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Exp([[INPUT]]) {
    // CHECK-SAME:          tilingStrategy = [1, 1, 1, 2]}
    // CHECK-SAME:      tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @SqrtSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
func.func @SqrtSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    %0 = VPU.Sqrt(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
    return %0 : tensor<1x8x80x960xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Sqrt([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @EluSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
func.func @EluSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    %0 = VPU.Elu(%arg0) {x = 1.000000e+00 : f64} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
    return %0 : tensor<1x8x80x960xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Elu([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2], x = 1.000000e+00 : f64} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @ClampSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
func.func @ClampSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    %0 = VPU.Clamp(%arg0) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
    return %0 : tensor<1x8x80x960xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Clamp([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @ReLUSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
func.func @ReLUSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    %0 = VPU.ReLU(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
    return %0 : tensor<1x8x80x960xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.ReLU([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @HSwishSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x16x80x960xf16>) -> tensor<1x16x80x960xf16>
func.func @HSwishSplitOverW(%arg0: tensor<1x16x80x960xf16>) -> tensor<1x16x80x960xf16> {
    %0 = VPU.HSwish(%arg0) : tensor<1x16x80x960xf16> -> tensor<1x16x80x960xf16>
    return %0 : tensor<1x16x80x960xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.HSwish([[INPUT]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 1, 4]} : tensor<1x16x80x960xf16> -> tensor<1x16x80x960xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x80x960xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @SplitDivideEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16>
func.func @SplitDivideEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
    %0 = VPU.Divide(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
    return %0 : tensor<1x10x256x176xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Divide([[INPUT_0]], [[INPUT_1]]) {
    // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @MemPermuteSplitNCHWToNHWC2Part
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x546x30x40xf16>) -> tensor<1x30x40x546xf16>
func.func @MemPermuteSplitNCHWToNHWC2Part(%arg0: tensor<1x546x30x40xf16>) -> tensor<1x30x40x546xf16> {
    %0 = VPU.MemPermute(%arg0) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>} : tensor<1x546x30x40xf16> -> tensor<1x30x40x546xf16>
    return %0 : tensor<1x30x40x546xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.MemPermute([[INPUT]]) {
    // CHECK-SAME:  dst_order = #NCHW, mem_perm = #NHWC, tilingStrategy = [1, 1, 1, 2]
    // CHECK-SAME:  } : tensor<1x546x30x40xf16> -> tensor<1x30x40x546xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x30x40x546xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @AvgPoolSwSplit2Part
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x24x1800x16xf16>) -> tensor<1x24x1789x16xf16>
func.func @AvgPoolSwSplit2Part(%arg0: tensor<1x24x1800x16xf16>) -> tensor<1x24x1789x16xf16> {
    %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x24x1800x16xf16> -> tensor<1x24x1789x16xf16>
    return %0 : tensor<1x24x1789x16xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.AvgPool([[INPUT]]) {
    // CHECK-SAME:  exclude_pads, kernel_size = [12, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1], tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:  } : tensor<1x24x1800x16xf16> -> tensor<1x24x1789x16xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x24x1789x16xf16>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL:   @SplitSparseNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x80x60xf16, {order = #NHWC}>
func.func @SplitSparseNCEConvOverOH(%arg0: tensor<1x32x80x60xf16, {order = #NHWC}>) -> tensor<1x160x80x60xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00> : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<160x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<160x1x1x384xi1>, is_weights>

    %0 = VPU.NCE.Convolution(%arg0, %weights_sparse) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [160, 32, 3, 3],
        strides = [1, 1]
    } : tensor<1x32x80x60xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<160x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<160x1x1x384xi1>, is_weights> -> tensor<1x160x80x60xf16, {order = #NHWC}>

    return %0 : tensor<1x160x80x60xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTS:%.+]] =  const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]

    // CHECK-DAG:        [[WEIGHTS_SM:%.+]] = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK:        [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights} -> !VPU.SparseTensor<
    // CHECK-SAME:       data=tensor<160x32x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:       sparsity_map=tensor<160x1x1x384xi1>, is_weights

    // CHECK:        [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SPARSE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [160, 32, 3, 3],
    // CHECK-SAME:          strides = [1, 1], tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:          -> tensor<1x160x80x60xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x160x80x60xf16, {order = #NHWC}>
}

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL:   @SplitSparseQuantNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x80x80x!qElemType, {order = #NHWC}>
func.func @SplitSparseQuantNCEConvOverOH(%arg0: tensor<1x32x80x80x!qElemType, {order = #NHWC}>) -> tensor<1x320x80x80x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<320x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<320x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<320x1x1x384xi1> = dense<1.000000e+00> : tensor<320x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<320x32x3x3x!qElemType2, {order = #NHWC}>, sparsity_map=tensor<320x1x1x384xi1>, is_weights>

    %0 = VPU.NCE.Convolution(%arg0, %weights_sparse) {
        ppe = #VPU.PPEStub<>,
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        rawFilterShape = [320, 32, 3, 3],
        strides = [1, 1]
    } : tensor<1x32x80x80x!qElemType, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<320x32x3x3x!qElemType2, {order = #NHWC}>, sparsity_map=tensor<320x1x1x384xi1>, is_weights> -> tensor<1x320x80x80x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x320x80x80x!qElemType1, {order = #NHWC}>

    // CHECK-DAG:        [[WEIGHTS:%.+]] = const.Declare tensor<320x32x3x3x!qElemType2, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<320x32x3x3xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!qElemType2>, #const.Reorder<#NHWC>, #const.Sparsify<false>]

    // CHECK-DAG:        [[WEIGHTS_SM:%.+]] = const.Declare tensor<320x1x1x384xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<320x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK:        [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights} -> !VPU.SparseTensor<
    // CHECK-SAME:       data=tensor<320x32x3x3x!qElemType2, {order = #NHWC}>,
    // CHECK-SAME:       sparsity_map=tensor<320x1x1x384xi1>, is_weights

    // CHECK:        [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SPARSE]])
    // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
    // CHECK-SAME:          rawFilterShape = [320, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:          -> tensor<1x320x80x80x!qElemType1, {order = #NHWC}>

    // CHECK:        return [[OUTPUT]] : tensor<1x320x80x80x!qElemType1, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @SplitNCEAveragePoolOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x7x12960xf16, {order = #NHWC}>
func.func @SplitNCEAveragePoolOverW(%arg0: tensor<1x16x7x12960xf16, {order = #NHWC}>) -> tensor<1x16x1x12960xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {kernel_size = [7, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x16x1x12960xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x12960xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {kernel_size = [7, 1]
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x16x1x12960xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x12960xf16, {order = #NHWC}>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @SplitAveragePoolOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1x7x184320xf16>
func.func @SplitAveragePoolOverW(%arg0: tensor<1x1x7x184320xf16>) -> tensor<1x1x1x184320xf16> {
    %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x7x184320xf16> -> tensor<1x1x1x184320xf16>

    return %0 : tensor<1x1x1x184320xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.AvgPool([[INPUT]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x1x1x184320xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x1x1x184320xf16>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL:   func.func @MVN1NormalizeSplit
func.func @MVN1NormalizeSplit(%arg0: tensor<1x1x1x520001xf16>, %arg1: tensor<1x1x1x2xf16, {order = #NHWC}>) -> tensor<1x1x1x520001xf16> {
    %0 = VPU.MVN1Normalize(%arg0, %arg1) {across_channels = false, normalize_variance = true} : tensor<1x1x1x520001xf16>, tensor<1x1x1x2xf16, {order = #NHWC}> -> tensor<1x1x1x520001xf16>
    return %0 : tensor<1x1x1x520001xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.MVN1Normalize(%arg0, %arg1)
    // CHECK-SAME:          tilingStrategy = [1, 1, 1, 2]
    // CHECK-SAME:     :  tensor<1x1x1x520001xf16>, tensor<1x1x1x2xf16, {order = #NHWC}> -> tensor<1x1x1x520001xf16>

    // CHECK:       return [[OUTPUT]] :  tensor<1x1x1x520001xf16>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL:   func.func @MVN1NormalizeSplitOverH
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x512x256x256xf16, {order = #NHWC}>, [[MEAN_VAR:%.+]]: tensor<1x512x1x32xf16, {order = #NHWC}>
func.func @MVN1NormalizeSplitOverH(%arg0: tensor<1x512x256x256xf16, {order = #NHWC}>, %arg1: tensor<1x512x1x32xf16, {order = #NHWC}>) -> tensor<1x512x256x256xf16, {order = #NHWC}> {
    %0 = VPU.MVN1Normalize(%arg0, %arg1) {across_channels = false, normalize_variance = true} : tensor<1x512x256x256xf16, {order = #NHWC}>, tensor<1x512x1x32xf16, {order = #NHWC}> -> tensor<1x512x256x256xf16, {order = #NHWC}>
    return %0 :  tensor<1x512x256x256xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.MVN1Normalize([[INPUT]], [[MEAN_VAR]])
    // CHECK-SAME:          tilingStrategy = [1, 1, 128, 1]
    // CHECK:       return [[OUTPUT]] : tensor<1x512x256x256xf16, {order = #NHWC}>
}

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

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

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 1 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

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
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

module @Test {

config.Resources 6 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @NCEMatMulSOGAndGTile
func.func @NCEMatMulSOGAndGTile(%arg0: tensor<64x8x64x32xf16>, %arg1: tensor<64x8x64x32xf16>) -> tensor<512x1x64x64x1xf16, {order = #GNHWC}> {
  %cst_0 = const.Declare tensor<512x64x1x1x4xsi32> = dense<10> : tensor<512x64x1x1x4xsi32>
  %0 = VPU.ShapeCast {shape = [1, 512, 64, 32]} inputs(%arg0 : tensor<64x8x64x32xf16>) -> tensor<1x512x64x32xf16>
  %1 = VPU.ShapeCast {shape = [1, 512, 64, 32]} inputs(%arg1 : tensor<64x8x64x32xf16>) -> tensor<1x512x64x32xf16>
  %2 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [512, 64, 32, 1, 1]} : tensor<1x512x64x32xf16> -> tensor<512x64x32x1x1xf16>
  %3 = VPU.PermuteCast(%2) {dst_order = #GNHWC, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>} : tensor<512x64x32x1x1xf16> -> tensor<512x1x32x64x1xf16, {order = #GNHWC}>
  %4 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [512, 64, 32, 1, 1]} : tensor<1x512x64x32xf16> -> tensor<512x64x32x1x1xf16>
  %5 = VPU.PermuteCast(%4) {dst_order = #GNHWC, mem_perm = #GNHWC} : tensor<512x64x32x1x1xf16> -> tensor<512x64x32x1x1xf16, {order = #GNHWC}>
  %6 = VPU.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [512, 1, 32, 16, 4]} : tensor<512x1x32x64x1xf16, {order = #GNHWC}> -> tensor<512x1x32x16x4xf16, {order = #GNHWC}>
  %7 = VPU.NCE.MatMul(%6, %5, %cst_0) {
    mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
    multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>,
    pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
    rawFilterShape = [512, 64, 32, 1, 1], strides = [1, 1], tilingStrategy = [2, 1, 1, 1, 1]} -> tensor<512x1x64x16x4xf16, {order = #GNHWC}>
  %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [512, 1, 64, 64, 1]} : tensor<512x1x64x16x4xf16, {order = #GNHWC}> -> tensor<512x1x64x64x1xf16, {order = #GNHWC}>
  return %8 : tensor<512x1x64x64x1xf16, {order = #GNHWC}>

  // CHECK:         VPU.NCE.MatMul
  // CHECK-SAME:    tilingStrategy = [2, 1, 1, 1, 1]
}
}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>

module @Test {

config.Resources 6 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: @NCEMatMulSOGAndHTile
func.func @NCEMatMulSOGAndHTile(%arg0: tensor<6x1x512x512xf16>, %arg1: tensor<6x1x512x512xf16>) -> tensor<6x1x512x512x1xf16, {order = #GNHWC}> {
  %cst = const.Declare tensor<6x512x1x1x4xsi32> = dense<10> : tensor<6x512x1x1x4xsi32>
  %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 6, 512, 512]} : tensor<6x1x512x512xf16> -> tensor<1x6x512x512xf16>
  %1 = VPU.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 6, 512, 512]} : tensor<6x1x512x512xf16> -> tensor<1x6x512x512xf16>
  %2 = VPU.AffineReshape(%0) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [6, 512, 512, 1, 1]} : tensor<1x6x512x512xf16> -> tensor<6x512x512x1x1xf16>
  %3 = VPU.PermuteCast(%2) {dst_order = #GNHWC, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>} : tensor<6x512x512x1x1xf16> -> tensor<6x1x512x512x1xf16, {order = #GNHWC}>
  %4 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [6, 512, 512, 1, 1]} : tensor<1x6x512x512xf16> -> tensor<6x512x512x1x1xf16>
  %5 = VPU.PermuteCast(%4) {dst_order = #GNHWC, mem_perm = #GNHWC} : tensor<6x512x512x1x1xf16> -> tensor<6x512x512x1x1xf16, {order = #GNHWC}>
  %6 = VPU.AffineReshape(%3) {dim_mapping = [[0], [1], [2], [3, 4], [4]], shape_value = [6, 1, 512, 128, 4]} : tensor<6x1x512x512x1xf16, {order = #GNHWC}> -> tensor<6x1x512x128x4xf16, {order = #GNHWC}>
  %7 = VPU.NCE.MatMul(%6, %5, %cst) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [6, 512, 512, 1, 1], strides = [1, 1]} -> tensor<6x1x512x128x4xf16, {order = #GNHWC}>
  %8 = VPU.AffineReshape(%7) {dim_mapping = [[0], [1], [2], [3], [3, 4]], shape_value = [6, 1, 512, 512, 1]} : tensor<6x1x512x128x4xf16, {order = #GNHWC}> -> tensor<6x1x512x512x1xf16, {order = #GNHWC}>
  return %8 : tensor<6x1x512x512x1xf16, {order = #GNHWC}>

  // CHECK:         VPU.NCE.MatMul
  // CHECK-SAME:    tilingStrategy = [1, 1, 1, 2, 1]
}
}

// -----

!qElemType = !quant.uniform<i4:f16, 1.000000e+00>

module @executors {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @DynamicDequantizeSplitOverH
    func.func @DynamicDequantizeSplitOverH(%arg0: tensor<1x32x1024x128x!qElemType>, %arg1: tensor<1x32x1024x1xf16>) -> tensor<1x32x1024x128xf16>{
        %0 = VPU.DynamicDequantize(%arg0, %arg1) {dstElemType = f16, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x32x1024x128x!qElemType>, tensor<1x32x1024x1xf16> -> tensor<1x32x1024x128xf16>
        return %0 : tensor<1x32x1024x128xf16>
    }

    // CHECK:       VPU.DynamicDequantize
    // CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]
}

// -----

// CHECK-LABEL: @ClampTilingNumForAlignment
!qElemType = !quant.uniform<i4:f32, 1.000000e+00>
module @ClampTilingNumForAlignment {
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input0" tensorNames = ["input0"] : tensor<1x32x128x11008xsi4>
    DataInfo "input1" tensorNames = ["input1"] : tensor<1x32x1x11008xf32>
  } outputsInfo : {
    DataInfo "output" tensorNames = ["output"] : tensor<1x32x128x11008xf16>
  }
  func.func @main( %arg16: tensor<1x32x128x11008xsi4>, %arg17: tensor<1x32x1x11008xf32>) -> (tensor<1x32x128x11008xf16>) {
    %552 = VPU.QuantizeCast(%arg16) {dstElemType = !qElemType} : tensor<1x32x128x11008xsi4> -> tensor<1x32x128x11008x!qElemType>
    %554 = VPU.Convert(%arg17) {dstElemType = f16} : tensor<1x32x1x11008xf32> -> tensor<1x32x1x11008xf16>
    %555 = VPU.DynamicDequantize(%552, %554) {dstElemType = f16} : tensor<1x32x128x11008x!qElemType>, tensor<1x32x1x11008xf16> -> tensor<1x32x128x11008xf16>
    return %555 : tensor<1x32x128x11008xf16>
    // CHECK:   func.func @main([[ARG0:%.+]]: tensor<1x32x128x11008xsi4>, [[ARG1:%.+]]: tensor<1x32x1x11008xf32>) -> tensor<1x32x128x11008xf16> {
        // CHECK:       [[QUANTIZECAST0:%.+]] = VPU.QuantizeCast([[ARG0]]) {dstElemType = !qElemType} : tensor<1x32x128x11008xsi4> -> tensor<1x32x128x11008x!qElemType>
        // CHECK:       [[CONVERT0:%.+]] = VPU.Convert([[ARG1]]) {dstElemType = f16, tilingStrategy = [1, 1, 1, 2]} : tensor<1x32x1x11008xf32> -> tensor<1x32x1x11008xf16>
        // CHECK:       [[DYNAMICDEQUANTIZE0:%.+]] = VPU.DynamicDequantize([[QUANTIZECAST0]], [[CONVERT0]]) {dstElemType = f16, tilingStrategy = [1, 2, 64, 1]} : tensor<1x32x128x11008x!qElemType>, tensor<1x32x1x11008xf16> -> tensor<1x32x128x11008xf16>
        // CHECK:       return [[DYNAMICDEQUANTIZE0]] : tensor<1x32x128x11008xf16>
  }
}

// -----

module @executors {
    config.Resources 1 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    // CHECK-LABEL:   @MultiplyNotAlign
    // CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x512x48x336xf16>,
    // CHECK-SAME:     [[INPUT1:%.+]]: tensor<1x512x48x336xf16>)
    func.func @MultiplyNotAlign(%arg0: tensor<1x512x48x336xf16>, %arg1: tensor<1x512x48x336xf16>) -> tensor<1x512x48x336xf16> {
        %0 = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
                    tensor<1x512x48x336xf16>, tensor<1x512x48x336xf16> -> tensor<1x512x48x336xf16>

        return %0 : tensor<1x512x48x336xf16>


        // CHECK: tilingStrategy = [1, 35, 1, 1]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @TilingForDWConvSEP {
    config.Resources 4 of @NCE at 6.000000e+02 MHz

// CHECK-LABEL: @DWConvWithSEPSOK
func.func @DWConvWithSEPSOK(%arg0: tensor<1x288x1x1xf16, {order = #NHWC}>) -> tensor<1x288x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<288x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<288x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %sparsity_map = const.Declare tensor<1x288x2x2xi1> = dense<1> : tensor<1x288x2x2xi1>

    %storage_element = VPU.StorageElementTable {
        dataElemType = f16,
        seDepth = 18, seSize = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
        dataShape = [1, 288, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 288, 2, 2]>
    } -> tensor<1x18x2x2xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 288, 2, 2]>
    } -> !VPU.SparseTensor<data=tensor<1x288x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x288x2x2xi1>,
                           storage_element_table=tensor<1x18x2x2xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 288, 2, 2]>>

    %interpolate = VPU.NCE.DepthConvolution(%input, %weights) {
        multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [288, 1, 1, 1],
        strides = [1, 1]
    } -> tensor<1x288x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x288x2x2xf16, {order = #NHWC}>

    // To satisfy DW.Conv + SEP requirements for workload channels, the op is
    // tiled into 2 slices, each with 144 channels; each individual op will then
    // be multiclustered with [64, 32, 32, 16] channels/cluster

    // CHECK:       VPU.NCE.DepthConvolution
    // CHECK-SAME:     tilingStrategy = [1, 2, 1, 1]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DWConvWithSEPNoMC
func.func @DWConvWithSEPNoMC(%arg0: tensor<1x288x1x1xf16, {order = #NHWC}>) -> tensor<1x288x2x2xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<288x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<288x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %sparsity_map = const.Declare tensor<1x288x2x2xi1> = dense<1> : tensor<1x288x2x2xi1>

    %storage_element = VPU.StorageElementTable {
        dataElemType = f16,
        seDepth = 18, seSize = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
        dataShape = [1, 288, 1, 1],
        seAttr = #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                    scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 288, 2, 2]>
    } -> tensor<1x18x2x2xi32, {order = #NHWC}>

    %input = VPU.GroupSparseTensor(%arg0, %sparsity_map, %storage_element) {
        seAttr = #VPU.SEInterpolate<
            mode = <NEAREST>,
            coordinate_transformation_mode = <ASYMMETRIC>,
            scale = [1.0, 1.0, 2.0, 2.0],
            nearest_mode = <FLOOR>,
            offsets = [0, 0, 0, 0],
            sizes = [1, 288, 2, 2]>
    } -> !VPU.SparseTensor<data=tensor<1x288x1x1xf16, {order = #NHWC}>,
                           sparsity_map=tensor<1x288x2x2xi1>,
                           storage_element_table=tensor<1x18x2x2xi32, {order = #NHWC}>,
                           #VPU.SEInterpolate<mode = <NEAREST>, coordinate_transformation_mode = <ASYMMETRIC>,
                                              scale = [1.0, 1.0, 2.0, 2.0], nearest_mode = <FLOOR>, offsets = [0, 0, 0, 0], sizes = [1, 288, 2, 2]>>

    %interpolate = VPU.NCE.DepthConvolution(%input, %weights) {
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [288, 1, 1, 1],
        strides = [1, 1]
    } -> tensor<1x288x2x2xf16, {order = #NHWC}>

    return %interpolate : tensor<1x288x2x2xf16, {order = #NHWC}>

    // To satisfy DW.Conv + SEP requirements for workload channels, the op is
    // tiled into 5 slices on channels, with division:
    // [64, 64, 64, 64, 32]

    // CHECK:       VPU.NCE.DepthConvolution
    // CHECK-SAME:     tilingStrategy = [1, 5, 1, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @Test {

config.Resources 6 of @NCE {
config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @DepthToSpaceDepthFirstSplitWithMultiCluster
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x320x576x672xf16, {order = #NHWC}>
func.func @DepthToSpaceDepthFirstSplitWithMultiCluster(%arg0: tensor<1x320x576x672xf16, {order = #NHWC}>) -> tensor<1x80x1152x1344xf16, {order = #NHWC}> {
    %0 = VPU.DepthToSpace(%arg0) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>} : tensor<1x320x576x672xf16, {order = #NHWC}> -> tensor<1x80x1152x1344xf16, {order = #NHWC}>

    return %0 : tensor<1x80x1152x1344xf16, {order = #NHWC}>

    // CHECK:       [[D2S:%.+]] = VPU.DepthToSpace([[INPUT]]) {block_size = 2 : i64, mode = #IE.depth_to_space_mode<DEPTH_FIRST>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, tilingStrategy = [1, 1, 96, 1]} : tensor<1x320x576x672xf16, {order = #NHWC}> -> tensor<1x80x1152x1344xf16, {order = #NHWC}>

    // CHECK:       return [[D2S]] : tensor<1x80x1152x1344xf16, {order = #NHWC}>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @AcoshOpSplitOverC
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1024x16x128xf16>
func.func @AcoshOpSplitOverC(%arg0: tensor<1x1024x16x128xf16>) -> tensor<1x1024x16x128xf16> {
    %0 = VPU.Acosh(%arg0) : tensor<1x1024x16x128xf16> -> tensor<1x1024x16x128xf16>

    return %0 : tensor<1x1024x16x128xf16>

    // CHECK:       [[ACOSH:%.+]] = VPU.Acosh([[INPUT]]) {
    // CHECK-SAME:          tilingStrategy = [1, 6, 1, 1]} : tensor<1x1024x16x128xf16> -> tensor<1x1024x16x128xf16>
    // CHECK:       return [[ACOSH]] : tensor<1x1024x16x128xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @AcosOpSplitOverC
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1024x16x128xf16>
func.func @AcosOpSplitOverC(%arg0: tensor<1x1024x16x128xf16>) -> tensor<1x1024x16x128xf16> {
    %0 = VPU.Acos(%arg0) : tensor<1x1024x16x128xf16> -> tensor<1x1024x16x128xf16>

    return %0 : tensor<1x1024x16x128xf16>

    // CHECK:       [[ACOS:%.+]] = VPU.Acos([[INPUT]]) {
    // CHECK-SAME:          tilingStrategy = [1, 6, 1, 1]} : tensor<1x1024x16x128xf16> -> tensor<1x1024x16x128xf16>
    // CHECK:       return [[ACOS]] : tensor<1x1024x16x128xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @AsinhOpSplitOverC
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1024x16x128xf16>
func.func @AsinhOpSplitOverC(%arg0: tensor<1x1024x16x128xf16>) -> tensor<1x1024x16x128xf16> {
    %0 = VPU.Asinh(%arg0) : tensor<1x1024x16x128xf16> -> tensor<1x1024x16x128xf16>

    return %0 : tensor<1x1024x16x128xf16>

    // CHECK:       [[ASINH:%.+]] = VPU.Asinh([[INPUT]]) {
    // CHECK-SAME:          tilingStrategy = [1, 6, 1, 1]} : tensor<1x1024x16x128xf16> -> tensor<1x1024x16x128xf16>
    // CHECK:       return [[ASINH]] : tensor<1x1024x16x128xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @AsinOpSplitOverC
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1024x16x128xf16>
func.func @AsinOpSplitOverC(%arg0: tensor<1x1024x16x128xf16>) -> tensor<1x1024x16x128xf16> {
    %0 = VPU.Asin(%arg0) : tensor<1x1024x16x128xf16> -> tensor<1x1024x16x128xf16>

    return %0 : tensor<1x1024x16x128xf16>

    // CHECK:       [[ASIN:%.+]] = VPU.Asin([[INPUT]]) {
    // CHECK-SAME:          tilingStrategy = [1, 6, 1, 1]} : tensor<1x1024x16x128xf16> -> tensor<1x1024x16x128xf16>
    // CHECK:       return [[ASIN]] : tensor<1x1024x16x128xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @AtanhOpSplitOverC
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1024x16x128xf16>
func.func @AtanhOpSplitOverC(%arg0: tensor<1x1024x16x128xf16>) -> tensor<1x1024x16x128xf16> {
    %0 = VPU.Atanh(%arg0) : tensor<1x1024x16x128xf16> -> tensor<1x1024x16x128xf16>

    return %0 : tensor<1x1024x16x128xf16>

    // CHECK:       [[ATANH:%.+]] = VPU.Atanh([[INPUT]]) {
    // CHECK-SAME:          tilingStrategy = [1, 6, 1, 1]} : tensor<1x1024x16x128xf16> -> tensor<1x1024x16x128xf16>
    // CHECK:       return [[ATANH]] : tensor<1x1024x16x128xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @AtanOpSplitOverC
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1024x16x128xf16>
func.func @AtanOpSplitOverC(%arg0: tensor<1x1024x16x128xf16>) -> tensor<1x1024x16x128xf16> {
    %0 = VPU.Atan(%arg0) : tensor<1x1024x16x128xf16> -> tensor<1x1024x16x128xf16>

    return %0 : tensor<1x1024x16x128xf16>

    // CHECK:       [[ATAN:%.+]] = VPU.Atan([[INPUT]]) {
    // CHECK-SAME:          tilingStrategy = [1, 6, 1, 1]} : tensor<1x1024x16x128xf16> -> tensor<1x1024x16x128xf16>
    // CHECK:       return [[ATAN]] : tensor<1x1024x16x128xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @SeluOpSplitOverC
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1024x16x128xf16>
func.func @SeluOpSplitOverC(%arg0: tensor<1x1024x16x128xf16>) -> tensor<1x1024x16x128xf16> {
    %0 = VPU.Selu(%arg0) {alpha_value = 1.000000e+00 : f64, lambda_value = 2.000000e+00 : f64} : tensor<1x1024x16x128xf16> -> tensor<1x1024x16x128xf16>

    return %0 : tensor<1x1024x16x128xf16>

    // CHECK:       [[SELU:%.+]] = VPU.Selu([[INPUT]]) {
    // CHECK-SAME:          alpha_value = 1.000000e+00 : f64, lambda_value = 2.000000e+00 : f64,
    // CHECK-SAME:          tilingStrategy = [1, 6, 1, 1]} : tensor<1x1024x16x128xf16> -> tensor<1x1024x16x128xf16>
    // CHECK:       return [[SELU]] : tensor<1x1024x16x128xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @CosOpSplitOverC
// CHECK-SAME:    [[INPUT:%.+]]: tensor<1x1024x16x128xf16>
func.func @CosOpSplitOverC(%arg0: tensor<1x1024x16x128xf16>) -> tensor<1x1024x16x128xf16> {
    %0 = VPU.Cos(%arg0) : tensor<1x1024x16x128xf16> -> tensor<1x1024x16x128xf16>

    return %0 : tensor<1x1024x16x128xf16>

    // CHECK:       [[COS:%.+]] = VPU.Cos([[INPUT]]) {
    // CHECK-SAME:          tilingStrategy = [1, 6, 1, 1]} : tensor<1x1024x16x128xf16> -> tensor<1x1024x16x128xf16>
    // CHECK:       return [[COS]] : tensor<1x1024x16x128xf16>
}

}

// -----

module @Test {

config.Resources 1 of @NCE {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}

// CHECK-LABEL: func.func @LSTMGatesSplitH
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x1x1536x2048xf16>,
// CHECK-SAME:    [[INPUT_0:%.+]]: tensor<1x1x1536x512xf16>)
func.func @LSTMGatesSplitH(%arg0: tensor<1x1x1536x2048xf16>, %arg1: tensor<1x1x1536x512xf16>) -> (tensor<1x1x1536x512xf16>, tensor<1x1x1536x512xf16>) {
    %0, %1 = VPU.LSTMGates(%arg0, %arg1) : tensor<1x1x1536x2048xf16>, tensor<1x1x1536x512xf16> -> tensor<1x1x1536x512xf16>, tensor<1x1x1536x512xf16>

    return %0, %1 : tensor<1x1x1536x512xf16>, tensor<1x1x1536x512xf16>

    // CHECK:       [[LSTMGATES_0:%.+]], [[LSTMGATES_1:%.+]] = VPU.LSTMGates([[INPUT]], [[INPUT_0]]) {
    // CHECK-SAME:          tilingStrategy = [1, 1, 8, 1]} : tensor<1x1x1536x2048xf16>, tensor<1x1x1536x512xf16> -> tensor<1x1x1536x512xf16>, tensor<1x1x1536x512xf16>

    // CHECK:       return [[LSTMGATES_0:%.+]], [[LSTMGATES_1:%.+]] : tensor<1x1x1536x512xf16>, tensor<1x1x1536x512xf16>
}

// CHECK-LABEL: func.func @AssignTilingStrategyAvgPool5D
// CHECK-SAME:          [[INPUT0:%arg[0-7]]]: tensor<1x24x8x56x56xf16>
func.func @AssignTilingStrategyAvgPool5D(%arg0: tensor<1x24x8x56x56xf16>) -> tensor<1x24x4x56x56xf16> {
    %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [2, 1, 1], pads_begin = [0, 0, 0], pads_end = [0, 0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 1, 1]} : tensor<1x24x8x56x56xf16> -> tensor<1x24x4x56x56xf16>
    return %0 : tensor<1x24x4x56x56xf16>

    // CHECK:               [[AVG_POOL:%.+]] = VPU.AvgPool([[INPUT0]]) {
    // CHECK-SAME:      exclude_pads, kernel_size = [2, 1, 1], pads_begin = [0, 0, 0], pads_end = [0, 0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [2, 1, 1], tilingStrategy = [1, 1, 1, 2, 1]} :
    // CHECK-SAME:               tensor<1x24x8x56x56xf16> -> tensor<1x24x4x56x56xf16>
    // CHECK:               return [[AVG_POOL]] : tensor<1x24x4x56x56xf16>
}

}


// -----

module @MultiplyWithSOK {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {VPU.bandwidth = 64 : i64, VPU.derateFactor = 1.000000e+00 : f64}
    }
    // CHECK-LABEL:   func.func @MultiplyWithSOK
    // CHECK-SAME:        [[INPUT:%.+]]: tensor<1x801x60x384xf16>
    func.func @MultiplyWithSOK(%arg0: tensor<1x801x60x384xf16>) -> tensor<1x801x60x384xf16> {
        %cst = const.Declare tensor<1x1x1x384xf16> = dense<0.200000e+00> : tensor<1x1x1x384xf16>
        %0 = VPU.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x801x60x384xf16>, tensor<1x1x1x384xf16> -> tensor<1x801x60x384xf16>

        return %0 : tensor<1x801x60x384xf16>
        // CHECK:       [[CST:%.+]] = const.Declare tensor<1x1x1x384xf16> = dense<1.999510e-01> : tensor<1x1x1x384xf16>
        // CHECK:       [[MULTIPLY:%.+]] = VPU.Multiply([[INPUT]], [[CST]]) {
        // CHECK-SAME:          auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
        // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        // CHECK-SAME:          tilingStrategy = [1, 48, 1, 1]} : tensor<1x801x60x384xf16>, tensor<1x1x1x384xf16> -> tensor<1x801x60x384xf16>
        // CHECK:       return [[MULTIPLY]] : tensor<1x801x60x384xf16>
    }
}
