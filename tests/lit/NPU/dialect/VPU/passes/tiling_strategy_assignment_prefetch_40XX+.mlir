//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --tiling-strategy-assignment %s | FileCheck %s
// REQUIRES: arch-NPU40XX

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: func.func @SplitSwConvOverOC
    // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x24x64x64xf16>,
    // CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<256x24x3x3xf16>,
    // CHECK-SAME:        [[BIAS:%arg[0-9]]]: tensor<1x256x1x1xf16>
    func.func @SplitSwConvOverOC(
            %input: tensor<1x24x64x64xf16>,
            %filter: tensor<256x24x3x3xf16>,
            %bias: tensor<1x256x1x1xf16>)
                -> tensor<1x256x64x64xf16> {
        %1 = VPU.Convolution(%input, %filter, %bias) {
            dilations = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            strides = [1, 1]
        } : tensor<1x24x64x64xf16>, tensor<256x24x3x3xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x64x64xf16>
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

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: func.func @SplitSwMaxPoolOverH
    // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x244x168xf16>
    func.func @SplitSwMaxPoolOverH(
            %input: tensor<1x16x244x168xf16>)
                -> tensor<1x16x244x168xf16> {
        %1 = VPU.MaxPool(%input) {
            kernel_size = [3, 3],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [1, 1]
        } : tensor<1x16x244x168xf16> -> tensor<1x16x244x168xf16>
        return %1 : tensor<1x16x244x168xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.MaxPool([[INPUT]])
        // CHECK-SAME:          kernel_size = [3, 3]
        // CHECK-SAME:          pads_begin = [1, 1]
        // CHECK-SAME:          pads_end = [1, 1]
        // CHECK-SAME:          rounding_type = #IE.rounding_type<FLOOR>
        // CHECK-SAME:          strides = [1, 1]
        // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
        // CHECK-SAME:      : tensor<1x16x244x168xf16> -> tensor<1x16x244x168xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x16x244x168xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: func.func @SplitSoftMaxWithSoK
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x4096x4096xf16>
    func.func @SplitSoftMaxWithSoK(%arg0: tensor<1x8x4096x4096xf16>) -> tensor<1x8x4096x4096xf16> {
        %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16>
        return %0 : tensor<1x8x4096x4096xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.SoftMax([[INPUT]]) {axisInd = 3 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, tilingStrategy = [1, 1, 94, 1]}
        // CHECK-SAME:      : tensor<1x8x4096x4096xf16> -> tensor<1x8x4096x4096xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x4096x4096xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: func.func @SplitSoftMaxOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x20x256x384xf16>
    func.func @SplitSoftMaxOverW(%arg0: tensor<1x20x256x384xf16>) -> tensor<1x20x256x384xf16> {
        %0 = VPU.SoftMax(%arg0) {axisInd = 1}: tensor<1x20x256x384xf16> -> tensor<1x20x256x384xf16>
        return %0 : tensor<1x20x256x384xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.SoftMax([[INPUT]]) {axisInd = 1 : i64, tilingStrategy = [1, 1, 6, 1]}
        // CHECK-SAME:      : tensor<1x20x256x384xf16> -> tensor<1x20x256x384xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x20x256x384xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: func.func @InterpSplitOverC
    // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x24x64x64xf16>
    func.func @InterpSplitOverC(
            %input1: tensor<1x24x64x64xf16>)
                -> tensor<1x24x256x256xf16> {

        %0 = VPU.Interpolate(%input1) {
                attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <LINEAR>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
                axes_attr = [2, 3], sizes_attr = [256, 256], operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0> } :
            tensor<1x24x64x64xf16> -> tensor<1x24x256x256xf16>

        return %0 : tensor<1x24x256x256xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Interpolate([[INPUT]]
        // CHECK-SAME:          pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
        // CHECK-SAME:          tilingStrategy = [1, 1, 3, 1]
        // CHECK-SAME:      :  tensor<1x24x64x64xf16>
        // CHECK-SAME:      -> tensor<1x24x256x256xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x24x256x256xf16>
    }

}

    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
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

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
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

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: func.func @SplitPReluOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>
    func.func @SplitPReluOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %cst = const.Declare tensor<1x8x1x1xf16> = dense<[-1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00]> : tensor<8xf32>, [#const.CastElemType<f16>, #const.Reshape<[1, 8, 1, 1]>]
        %0 = VPU.PRelu(%arg0, %cst) : tensor<1x8x80x960xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x8x1x1xf16> = dense<[-1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00]>

        // CHECK:       [[OUTPUT:%.+]] = VPU.PRelu([[INPUT]], [[CST]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: func.func @SplitLeakyReluOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>
    func.func @SplitLeakyReluOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.LeakyRelu(%arg0) {negative_slope = 0.0099999997764825821 : f64} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.LeakyRelu([[INPUT]]) {
        // CHECK-SAME:  negative_slope = 0.0099999997764825821 : f64, tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: func.func @GenericTiling
    // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x144x20x20xf16, {order = #NHWC}>,
    // CHECK-SAME:        [[WEIGHTS1:%arg[0-9]]]: tensor<144x144x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:        [[WEIGHTS2:%arg[0-9]]]: tensor<576x144x3x3xf16, {order = #NHWC}>
    func.func @GenericTiling(
            %input: tensor<1x144x20x20xf16, {order = #NHWC}>,
            %weights1: tensor<144x144x3x3xf16, {order = #NHWC}>,
            %weights2: tensor<576x144x3x3xf16, {order = #NHWC}>)
                -> tensor<1x576x20x20xf16, {order = #NHWC}> {
        %1 = VPU.NCE.Convolution(%input, %weights1) {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [144, 144, 3, 3],
            strides = [1, 1]
        } : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<144x144x3x3xf16, {order = #NHWC}> -> tensor<1x144x20x20xf16, {order = #NHWC}>
        %2 = VPU.NCE.Eltwise(%1, %1) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<1x144x20x20xf16, {order = #NHWC}> -> tensor<1x144x20x20xf16, {order = #NHWC}>
        %3 = VPU.NCE.Convolution(%2, %weights2) {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [576, 144, 3, 3],
            strides = [1, 1]
        } : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<576x144x3x3xf16, {order = #NHWC}> -> tensor<1x576x20x20xf16, {order = #NHWC}>
        return %3 : tensor<1x576x20x20xf16, {order = #NHWC}>

        // CHECK:       [[CONV_1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]])
        // CHECK-SAME:     {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [144, 144, 3, 3], strides = [1, 1]}
        // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>

        // CHECK:       [[AND:%.+]] = VPU.NCE.Eltwise([[CONV_1]], [[CONV_1]]) {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
        // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>

        // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution([[AND]], [[WEIGHTS2]])
        // CHECK-SAME:     {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEStub<>, rawFilterShape = [576, 144, 3, 3], strides = [1, 1], tilingStrategy = [1, 4, 1, 1]}
        // CHECK-SAME:          -> tensor<1x576x20x20xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x576x20x20xf16, {order = #NHWC}>
    }

}

    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @NoTileWithSOH
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x100x100xf16, {order = #NHWC}>
    func.func @NoTileWithSOH(
            %arg0: tensor<1x32x100x100xf16, {order = #NHWC}>)
                -> tensor<1x128x100x100xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
            : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>]

        %0 = VPU.NCE.Convolution(%arg0, %weights) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [128, 32, 3, 3],
            strides = [1, 1]
        } : tensor<1x32x100x100xf16, {order = #NHWC}>, tensor<128x32x3x3xf16, {order = #NHWC}> -> tensor<1x128x100x100xf16, {order = #NHWC}>

        return %0 : tensor<1x128x100x100xf16, {order = #NHWC}>

        // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}>

        // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]])
        // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
        // CHECK-SAME:          rawFilterShape = [128, 32, 3, 3]
        // CHECK-SAME:          strides = [1, 1]
        // CHECK-NOT:           tilingStrategy
        // CHECK-SAME:          tensor<1x128x100x100xf16, {order = #NHWC}>

        // CHECK:       return [[CONV]] : tensor<1x128x100x100xf16, {order = #NHWC}>
    }

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @TileWithSOH
    // CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x16x210x512xf16, {order = #NHWC}>
    func.func @TileWithSOH(
            %arg0: tensor<1x16x210x512xf16, {order = #NHWC}>)
                -> tensor<1x32x210x512xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
            : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]

        %0 = VPU.NCE.Convolution(%arg0, %weights) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [32, 16, 3, 3],
            strides = [1, 1]
        } : tensor<1x16x210x512xf16, {order = #NHWC}>, tensor<32x16x3x3xf16, {order = #NHWC}> -> tensor<1x32x210x512xf16, {order = #NHWC}>

        return %0 : tensor<1x32x210x512xf16, {order = #NHWC}>

        // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}>

        // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]])
        // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>
        // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
        // CHECK-SAME:          rawFilterShape = [32, 16, 3, 3]
        // CHECK-SAME:          tensor<1x32x210x512xf16, {order = #NHWC}>

        // CHECK:       return [[CONV1]] : tensor<1x32x210x512xf16, {order = #NHWC}>
    }

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @NoTileWithSOK
    // CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x10x10xf16, {order = #NHWC}>
    func.func @NoTileWithSOK(
            %arg0: tensor<1x32x10x10xf16, {order = #NHWC}>)
                -> tensor<1x240x10x10xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<240x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00>
            : tensor<240x32x7x7xf16>, [#const.Reorder<#NHWC>]

        %0 = VPU.NCE.Convolution(%arg0, %weights) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [240, 32, 7, 7],
            strides = [1, 1]
        } : tensor<1x32x10x10xf16, {order = #NHWC}>, tensor<240x32x7x7xf16, {order = #NHWC}> -> tensor<1x240x10x10xf16, {order = #NHWC}>

        return %0 : tensor<1x240x10x10xf16, {order = #NHWC}>

        // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<240x32x7x7xf16, {order = #NHWC}>

        // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]])
        // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
        // CHECK-SAME:          pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
        // CHECK-SAME:          rawFilterShape = [240, 32, 7, 7],
        // CHECK-SAME:          strides = [1, 1]
        // CHECK-NOT:           tilingStrategy
        // CHECK-SAME:          tensor<1x240x10x10xf16, {order = #NHWC}>

        // CHECK:       return [[CONV]] : tensor<1x240x10x10xf16, {order = #NHWC}>
    }

}

    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @LargeConstPipeliningSOKFor
    // CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x256x14x14xf16, {order = #NHWC}>
    func.func @LargeConstPipeliningSOKFor(
            %arg0: tensor<1x256x14x14xf16, {order = #NHWC}>)
                -> tensor<1x512x14x14xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<512x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
            : tensor<512x256x3x3xf16>, [#const.Reorder<#NHWC>]

        %0 = VPU.NCE.Convolution(%arg0, %weights) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [512, 256, 3, 3],
            strides = [1, 1]
        } : tensor<1x256x14x14xf16, {order = #NHWC}>, tensor<512x256x3x3xf16, {order = #NHWC}> -> tensor<1x512x14x14xf16, {order = #NHWC}>

        return %0 : tensor<1x512x14x14xf16, {order = #NHWC}>

        // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<512x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        // CHECK-SAME:          : tensor<512x256x3x3xf16>, [#const.Reorder<#NHWC>]

        // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]])
        // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
        // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
        // CHECK-SAME:          rawFilterShape = [512, 256, 3, 3]
        // CHECK-SAME:          tilingStrategy = [1, 2, 1, 1]
        // CHECK-SAME:          -> tensor<1x512x14x14xf16, {order = #NHWC}>

        // CHECK:       return [[CONV1]] : tensor<1x512x14x14xf16, {order = #NHWC}>
    }

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: func.func @SplitNCEEltwise
    // CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<1x512x28x28xf16, {order = #NHWC}>,
    // CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<1x512x28x28xf16, {order = #NHWC}>
    func.func @SplitNCEEltwise(
            %arg0: tensor<1x512x28x28xf16, {order = #NHWC}>,
            %arg1: tensor<1x512x28x28xf16, {order = #NHWC}>)
                -> tensor<1x512x28x28xf16, {order = #NHWC}> {
        %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
            ppe = #VPU.PPEStub<>,
            op_type = #VPU.eltwise_type<ADD>
        } -> tensor<1x512x28x28xf16, {order = #NHWC}>

        return %0 : tensor<1x512x28x28xf16, {order = #NHWC}>

        // CHECK:       [[ELTWISE_0:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]])
        // CHECK-SAME:      {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>, tilingStrategy = [1, 2, 1, 1]}
        // CHECK-SAME:      -> tensor<1x512x28x28xf16, {order = #NHWC}>

        // return [[ELTWISE_0]] : tensor<1x512x28x28xf16, {order = #NHWC}>
    }

}
    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: func.func @NoPrefetchingForEltwise
    // CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<1x32x70x50xf16, {order = #NHWC}>,
    // CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<1x64x70x50xf16, {order = #NHWC}>
    func.func @NoPrefetchingForEltwise(
            %arg0: tensor<1x32x70x50xf16, {order = #NHWC}>,
            %arg1: tensor<1x64x70x50xf16, {order = #NHWC}>)
                -> tensor<1x64x70x50xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]

        %0 = VPU.NCE.Convolution(%arg0, %weights) {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [64, 32, 3, 3],
            strides = [1, 1]
        } : tensor<1x32x70x50xf16, {order = #NHWC}>, tensor<64x32x3x3xf16, {order = #NHWC}> -> tensor<1x64x70x50xf16, {order = #NHWC}>

        %1 = VPU.NCE.Eltwise(%0, %arg1) {
            ppe = #VPU.PPEStub<>,
            op_type = #VPU.eltwise_type<ADD>
        } -> tensor<1x64x70x50xf16, {order = #NHWC}>

        return %1 : tensor<1x64x70x50xf16, {order = #NHWC}>

        // CHECK-DAG:       [[WEIGHTS:%.+]]       = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>

        // CHECK:       [[PARENT_CONV:%.+]] = VPU.NCE.Convolution([[INPUT_0]], [[WEIGHTS]])
        // CHECK-SAME:          -> tensor<1x64x70x50xf16, {order = #NHWC}>

        // Eltwise is not tiled for prefetching
        // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[PARENT_CONV]], [[INPUT_1]])
        // CHECK-SAME:              op_type = #VPU.eltwise_type<ADD>
        // CHECK-NOT:               tilingStrategy
        // CHECK-SAME:          -> tensor<1x64x70x50xf16, {order = #NHWC}>

        // return [[ELTWISE]] : tensor<1x64x70x50xf16, {order = #NHWC}>
    }

}

    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitAveragePoolOverW
    // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1x7x184320xf16>
    func.func @SplitAveragePoolOverW(%arg0: tensor<1x1x7x184320xf16>) -> tensor<1x1x1x184320xf16> {
        %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x7x184320xf16> -> tensor<1x1x1x184320xf16>

        return %0 : tensor<1x1x1x184320xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.AvgPool([[INPUT]])
        // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]
        // CHECK-SAME:      -> tensor<1x1x1x184320xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x1x1x184320xf16>
    }

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 1 of @NCE {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL:   func.func @MVN1NormalizeSplit
    // CHECK-SAME:        [[INPUT1:%.+]]: tensor<1x1x1x520001xf16>
    // CHECK-SAME:        [[INPUT2:%.+]]: tensor<1x1x1x2xf16, {order = #NHWC}>
    func.func @MVN1NormalizeSplit(%input1: tensor<1x1x1x520001xf16>, %input2: tensor<1x1x1x2xf16, {order = #NHWC}>) -> tensor<1x1x1x520001xf16> {
        %0 = VPU.MVN1Normalize(%input1, %input2) {across_channels = false, normalize_variance = true} : tensor<1x1x1x520001xf16>, tensor<1x1x1x2xf16, {order = #NHWC}> -> tensor<1x1x1x520001xf16>
        return %0 : tensor<1x1x1x520001xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.MVN1Normalize([[INPUT1]], [[INPUT2]])
        // CHECK-SAME:          tilingStrategy = [1, 1, 1, 3]
        // CHECK-SAME:     :  tensor<1x1x1x520001xf16>, tensor<1x1x1x2xf16, {order = #NHWC}> -> tensor<1x1x1x520001xf16>

        // CHECK:       return [[OUTPUT]] :  tensor<1x1x1x520001xf16>
    }

}

    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL:   func.func @MVN1NormalizeSplitOverH
    // CHECK-SAME:    [[INPUT:%.+]]: tensor<1x512x256x256xf16, {order = #NHWC}>, [[MEAN_VAR:%.+]]: tensor<1x512x1x32xf16, {order = #NHWC}>
    func.func @MVN1NormalizeSplitOverH(%arg0: tensor<1x512x256x256xf16, {order = #NHWC}>, %arg1: tensor<1x512x1x32xf16, {order = #NHWC}>) -> tensor<1x512x256x256xf16, {order = #NHWC}> {
        %0 = VPU.MVN1Normalize(%arg0, %arg1) {across_channels = false, normalize_variance = true} : tensor<1x512x256x256xf16, {order = #NHWC}>, tensor<1x512x1x32xf16, {order = #NHWC}> -> tensor<1x512x256x256xf16, {order = #NHWC}>
        return %0 :  tensor<1x512x256x256xf16, {order = #NHWC}>

        // CHECK:       [[OUTPUT:%.+]] = VPU.MVN1Normalize([[INPUT]], [[MEAN_VAR]])
        // CHECK-SAME:          tilingStrategy = [1, 1, 256, 1]
        // CHECK:       return [[OUTPUT]] : tensor<1x512x256x256xf16, {order = #NHWC}>
    }

}

    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @ClampSplitOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    func.func @ClampSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Clamp(%arg0) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Clamp([[INPUT]]) {
        // CHECK-SAME:  max = 1.000000e+00 : f64, min = -1.000000e+00 : f64, tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @ReLUSplitOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    func.func @ReLUSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.ReLU(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.ReLU([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

    #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @LogSplitOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    func.func @LogSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Log(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Log([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @AbsSplitOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    func.func @AbsSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Abs(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Abs([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitFloorModEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16>
    func.func @SplitFloorModEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
        %0 = VPU.FloorMod(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        return %0 : tensor<1x10x256x176xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.FloorMod([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitModEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16>
    func.func @SplitModEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
        %0 = VPU.Mod(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        return %0 : tensor<1x10x256x176xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Mod([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitPowerEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16>
    func.func @SplitPowerEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
        %0 = VPU.Power(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        return %0 : tensor<1x10x256x176xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Power([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitLogicalOrEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16>
    func.func @SplitLogicalOrEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
        %0 = VPU.LogicalOr(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        return %0 : tensor<1x10x256x176xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.LogicalOr([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitLogicalXorEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16>
    func.func @SplitLogicalXorEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
        %0 = VPU.LogicalXor(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        return %0 : tensor<1x10x256x176xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.LogicalXor([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitEqualEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8>
    func.func @SplitEqualEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {
        %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>
        return %0 : tensor<1x10x256x176xi8>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Equal([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xi8>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitNotEqualEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8>
    func.func @SplitNotEqualEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {
        %0 = VPU.NotEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>
        return %0 : tensor<1x10x256x176xi8>

        // CHECK:       [[OUTPUT:%.+]] = VPU.NotEqual([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xi8>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitLessEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8>
    func.func @SplitLessEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {
        %0 = VPU.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>
        return %0 : tensor<1x10x256x176xi8>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Less([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xi8>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitLessEqualEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8>
    func.func @SplitLessEqualEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {
        %0 = VPU.LessEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>
        return %0 : tensor<1x10x256x176xi8>

        // CHECK:       [[OUTPUT:%.+]] = VPU.LessEqual([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xi8>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitGreaterEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8>
    func.func @SplitGreaterEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xi8> {
        %0 = VPU.Greater(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>
        return %0 : tensor<1x10x256x176xi8>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Greater([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xi8>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xi8>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitGreaterEqualEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16>
    func.func @SplitGreaterEqualEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
        %0 = VPU.GreaterEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        return %0 : tensor<1x10x256x176xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.GreaterEqual([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
    }

}
    // -----

    #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitErfOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    func.func @SplitErfOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Erf(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Erf([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitFloorOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    func.func @SplitFloorOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Floor(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Floor([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

    #NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @TanSplitOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    func.func @TanSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Tan(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Tan([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SwishSplitOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    func.func @SwishSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Swish(%arg0) {beta_value = 1.000000e+00 : f64} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Swish([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @HSigmoidSplitOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
    func.func @HSigmoidSplitOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.HSigmoid(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.HSigmoid([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitNegativeActivationSw
    // CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
    func.func @SplitNegativeActivationSw(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Negative(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Negative(%arg0) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitCeilingActivationSw
    // CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
    func.func @SplitCeilingActivationSw(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Ceiling(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Ceiling([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitSignActivationSw
    // CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
    func.func @SplitSignActivationSw(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Sign(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Sign([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}

    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
        config.ExecutorResource 2 of @SHAVE_ACT
    }

    // CHECK-LABEL: @SplitSelectEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_2:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
    func.func @SplitSelectEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>, %arg2: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
        %0 = VPU.Select(%arg0, %arg1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        return %0 : tensor<1x10x256x176xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Select([[INPUT_0]], [[INPUT_1]], [[INPUT_2]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 5, 1, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
    }

}

    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitAddEltwiseSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x176xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16>
    func.func @SplitAddEltwiseSw(%arg0: tensor<1x10x256x176xf16>, %arg1: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
        %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        return %0 : tensor<1x10x256x176xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Add([[INPUT_0]], [[INPUT_1]]) {
        // CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>, tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16>, tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitRoundActivationSw
    // CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
    func.func @SplitRoundActivationSw(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Round([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitGeluActivationSw
    // CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16>
    func.func @SplitGeluActivationSw(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x960xf16> {
        %0 = VPU.Gelu(%arg0) : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>
        return %0 : tensor<1x8x80x960xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.Gelu([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x960xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x960xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitTopK
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x5x512x384xf16>) -> (tensor<1x1x512x384xf32>, tensor<1x1x512x384xsi32>)
    func.func @SplitTopK(%arg0: tensor<1x5x512x384xf16>) -> (tensor<1x1x512x384xf32>, tensor<1x1x512x384xsi32>) {
        %cst = const.Declare tensor<1x1x1x16xui8> = dense<0> : tensor<1x1x1x16xui8>
        %output_values, %target_shape = VPU.TopK(%arg0, %cst) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, operandSegmentSizes = array<i32: 1, 0, 1>, sort = #IE.topk_sort_type<SORT_INDICES>} : tensor<1x5x512x384xf16>, tensor<1x1x1x16xui8> -> tensor<1x1x512x384xf16>, tensor<1x1x512x384xsi32>
        %0 = VPU.Convert(%output_values) {dstElemType = f32} : tensor<1x1x512x384xf16> -> tensor<1x1x512x384xf32>
        return %0, %target_shape : tensor<1x1x512x384xf32>, tensor<1x1x512x384xsi32>

        // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x1x16xui8> = dense<0> : tensor<1x1x1x16xui8>
        // CHECK: [[OUTPUT_VALUE:%.+]], [[TARGET_SHAPE:%.+]] = VPU.TopK([[INPUT_0]], [[CST]]) {axis = 1 : i64, element_type = si32, k_value = 1 : i64, mode = #IE.topk_mode<MAX>, operandSegmentSizes = array<i32: 1, 0, 1>, sort = #IE.topk_sort_type<SORT_INDICES>, tilingStrategy = [1, 1, 3, 1]} : tensor<1x5x512x384xf16>, tensor<1x1x1x16xui8> -> tensor<1x1x512x384xf16>, tensor<1x1x512x384xsi32>
        // CHECK: [[OUTPUT_VALUE_CONV:%.+]] = VPU.Convert([[OUTPUT_VALUE]]) {dstElemType = f32} : tensor<1x1x512x384xf16> -> tensor<1x1x512x384xf32>
        // CHECK: return [[OUTPUT_VALUE_CONV]], [[TARGET_SHAPE]] : tensor<1x1x512x384xf32>, tensor<1x1x512x384xsi32>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: func.func @SplitStridedSliceOverW
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x960xf16>
    func.func @SplitStridedSliceOverW(%arg0: tensor<1x8x80x960xf16>) -> tensor<1x8x80x480xf16> {
        %0 = VPU.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 8, 80, 960], new_axis_mask = [0, 0, 0, 0], operandSegmentSizes = array<i32: 1, 0, 0, 0>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x480xf16>
        return %0 : tensor<1x8x80x480xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.StridedSlice([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 2]} : tensor<1x8x80x960xf16> -> tensor<1x8x80x480xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x8x80x480xf16>
    }

}
    // -----

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitLogicalNotEltwiseSw
    // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16>
    func.func @SplitLogicalNotEltwiseSw(%arg0: tensor<1x10x256x176xf16>) -> tensor<1x10x256x176xf16> {
        %0 = VPU.LogicalNot(%arg0) : tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>
        return %0 : tensor<1x10x256x176xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.LogicalNot([[INPUT]]) {
        // CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x256x176xf16> -> tensor<1x10x256x176xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x10x256x176xf16>
    }

}

    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

    !qElemType = !quant.uniform<u8:f16, 1.0>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: func.func @PrefetchTilingWithParentConsidered
    // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x512x14x14x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:        [[WEIGHTS1:%arg[0-9]]]: tensor<512x512x3x3x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:        [[WEIGHTS2:%arg[0-9]]]: tensor<2048x512x1x1x!qElemType, {order = #NHWC}>
    func.func @PrefetchTilingWithParentConsidered(
            %input: tensor<1x512x14x14x!qElemType, {order = #NHWC}>,
            %weights1: tensor<512x512x3x3x!qElemType, {order = #NHWC}>,
            %weights2: tensor<2048x512x1x1x!qElemType, {order = #NHWC}>)
                -> tensor<1x2048x7x7x!qElemType, {order = #NHWC}> {
        %0 = VPU.NCE.Convolution(%input, %weights1) {
            pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [512, 512, 3, 3], strides = [2, 2]}
                : tensor<1x512x14x14x!qElemType, {order = #NHWC}>, tensor<512x512x3x3x!qElemType, {order = #NHWC}> -> tensor<1x512x7x7x!qElemType, {order = #NHWC}>
        %1 = VPU.NCE.Convolution(%0, %weights2) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [2048, 512, 1, 1], strides = [1, 1]}
                : tensor<1x512x7x7x!qElemType, {order = #NHWC}>, tensor<2048x512x1x1x!qElemType, {order = #NHWC}> -> tensor<1x2048x7x7x!qElemType, {order = #NHWC}>
        return %1 : tensor<1x2048x7x7x!qElemType, {order = #NHWC}>

        // Prefetching mode is triggered for the child conv
        // with tiled parent memory considered
        // CHECK:       [[PARENT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]])
        // CHECK-SAME:  tilingStrategy = [1, 2, 1, 1]
        // CHECK:       [[CHILD:%.+]] = VPU.NCE.Convolution([[PARENT]], [[WEIGHTS2]])
        // CHECK-SAME:  tilingStrategy = [1, 64, 1, 1]
    }

}

// -----

    !qElemType = !quant.uniform<i4:f16, 1.000000e+00>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: func.func @SplitDequantizeWithSoH
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x28x4608x128x!qElemType>,
    // CHECK-SAME:  [[SCALE:%arg[0-9]]]: tensor<1x28x4608x1xf16>
    func.func @SplitDequantizeWithSoH(%arg0: tensor<1x28x4608x128x!qElemType>, %arg1: tensor<1x28x4608x1xf16>) -> tensor<1x28x4608x128xf16> {
        %0 = VPU.DynamicDequantize(%arg0, %arg1) {
            dstElemType = f16,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
        } : tensor<1x28x4608x128x!qElemType>, tensor<1x28x4608x1xf16> -> tensor<1x28x4608x128xf16>

        return %0 : tensor<1x28x4608x128xf16>

        // CHECK:       [[OUTPUT:%.+]] = VPU.DynamicDequantize([[INPUT]], [[SCALE]]) {
        // CHECK-SAME:      dstElemType = f16,
        // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        // CHECK-SAME:      tilingStrategy = [1, 1, 6, 1]
        // CHECK-SAME:      } : tensor<1x28x4608x128x!qElemType>, tensor<1x28x4608x1xf16> -> tensor<1x28x4608x128xf16>

        // CHECK:       return [[OUTPUT]] : tensor<1x28x4608x128xf16>
    }
}

    // -----

    #NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: func.func @SplitPReluOverH
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x64x10240x1xf16, {order = #NHWC}>
    func.func @SplitPReluOverH(%arg0: tensor<1x64x10240x1xf16, {order = #NHWC}>) -> tensor<1x64x10240x1xf16, {order = #NHWC}> {
        %cst = const.Declare tensor<1x64x1x1xf16, {order = #NHWC}> =
                dense<1.000000e+00> : tensor<1x64x1x1xf16>, [#const.Reorder<#NHWC>]
        %0 = VPU.PRelu(%arg0, %cst) : tensor<1x64x10240x1xf16, {order = #NHWC}>, tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x10240x1xf16, {order = #NHWC}>
        return %0 : tensor<1x64x10240x1xf16, {order = #NHWC}>


        // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<1x64x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x64x1x1xf16>, [#const.Reorder<#NHWC>]
        // CHECK:       [[OUTPUT:%.+]] = VPU.PRelu([[INPUT]], [[CST]]) {tilingStrategy = [1, 1, 2, 1]} : tensor<1x64x10240x1xf16, {order = #NHWC}>, tensor<1x64x1x1xf16, {order = #NHWC}> -> tensor<1x64x10240x1xf16, {order = #NHWC}>
        // CHECK:       return [[OUTPUT]] : tensor<1x64x10240x1xf16, {order = #NHWC}>
    }

}

// -----

#GNHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#GNCHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!qElemType = !quant.uniform<u8:f16, 0.0016544117647058823>
module @executors {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
// CHECK-LABEL: @NCEMatMulSOGAndPipelined
  func.func @NCEMatMulSOGAndPipelined(%arg0: tensor<1x32x4x32xf16, {order = #NHWC}>) ->  tensor<32x1x1408x1x1xf16, {order = #GNHWC}>{
    %weight_depth_conv = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<0.0> : tensor<32x16x1x1xf16, {order = #NHWC}>

    %depth_conv = VPU.NCE.DepthConvolution(%arg0, %weight_depth_conv) {multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1, 1, 1], strides = [1, 1]} -> tensor<1x32x4x32xf16, {order = #NHWC}>
    %31 = VPU.ShapeCast {shape = [1, 128, 32, 1]} inputs(%depth_conv : tensor<1x32x4x32xf16, {order = #NHWC}>) -> tensor<1x128x32x1xf16, {order = #NHWC}> loc(fused<{name = "Multiply_72943", type = "Multiply"}>["Multiply_72943"])
    %32 = VPU.PermuteCast(%31) {dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>} : tensor<1x128x32x1xf16, {order = #NHWC}> -> tensor<1x32x1x128xf16> loc(fused<{name = "Multiply_72943", type = "Multiply"}>["Multiply_72943", "reorder_out"])
    %47 = VPU.AffineReshape(%32) {dim_mapping = [[0], [0], [1], [2, 3, 4]], shape_value = [32, 1, 128, 1, 1]} : tensor<1x32x1x128xf16> -> tensor<32x1x128x1x1xf16> loc(fused<{name = "Multiply_72943", type = "Multiply"}>["Multiply_72943", "reorder_out"])
    %48 = VPU.PermuteCast(%47) {dst_order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>, mem_perm = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4, d2)>} : tensor<32x1x128x1x1xf16> -> tensor<32x1x128x1x1xf16, {order = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>}> loc(fused<{name = "Multiply_72943", type = "Multiply"}>["Multiply_72943", "reorder_out"])

    %weight = const.Declare tensor<32x1408x128x1x1xf16, {order = #GNHWC}> = dense<10.0> : tensor<32x1408x128x1x1xf16, {order = #GNHWC}>
    %weight_table = const.Declare tensor<32x1408x1x1x4xsi32> = dense<0> : tensor<32x1408x1x1x4xsi32>
    %grouped_matmul = VPU.NCE.MatMul(%48, %weight, %weight_table) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverGroup>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 1408, 128, 1, 1], strides = [1, 1]} -> tensor<32x1x1408x1x1xf16, {order = #GNHWC}>

    return %grouped_matmul : tensor<32x1x1408x1x1xf16, {order = #GNHWC}>
    // CHECK:         VPU.NCE.MatMul
    // CHECK-SAME:    tilingStrategy = [8, 1, 1, 1, 1]
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

    !quantileFloatType = !QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>
    !qElemType = !quant.quantile<u4:f16:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:1.000000e+00>

module @executors {
    config.Resources 4 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: func.func @SplitConvWithLargeFilter
    func.func @SplitConvWithLargeFilter(%arg0: tensor<1x3840x1x1xf16, {order = #NHWC}>, %arg1: tensor<1536x3840x!quantileFloatType>) -> tensor<1x1536x1x1xf16, {order = #NHWC}> {
        %0 = VPU.QuantizeCast(%arg1) {dstElemType = !qElemType} : tensor<1536x3840x!quantileFloatType> -> tensor<1536x3840x!qElemType>
        %1 = VPU.AffineReshape(%0) {dim_mapping = [[0], [1, 2, 3]], shape_value = [1536, 3840, 1, 1]} : tensor<1536x3840x!qElemType> -> tensor<1536x3840x1x1x!qElemType>
        %2 = VPU.PermuteCast(%1) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1536x3840x1x1x!qElemType> -> tensor<1536x3840x1x1x!qElemType, {order = #NHWC}>
        %3 = VPU.NCE.Convolution(%arg0, %2) {
            mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            rawFilterShape = [1536, 3840, 1, 1],
            strides = [1, 1]
        } : tensor<1x3840x1x1xf16, {order = #NHWC}>, tensor<1536x3840x1x1x!qElemType, {order = #NHWC}> -> tensor<1x1536x1x1xf16, {order = #NHWC}>

        return %3 : tensor<1x1536x1x1xf16, {order = #NHWC}>

        // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Convolution
        // CHECK-SAME:      mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
        // CHECK-SAME:      multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
        // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        // CHECK-SAME:      ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
        // CHECK-SAME:      rawFilterShape = [1536, 3840, 1, 1],
        // CHECK-SAME:      strides = [1, 1],
        // CHECK-SAME:      tilingStrategy = [1, 2, 1, 1]
        // CHECK-SAME:  }
        // CHECK-SAME:  -> tensor<1x1536x1x1xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x1536x1x1xf16, {order = #NHWC}>
    }
}
