//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --tiling-strategy-assignment %s | FileCheck %s
// REQUIRES: arch-NPU50XX

module @executors {
    config.Resources 3 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }
    // CHECK-LABEL:   @MultiplyPipeliningTiling
    // CHECK-SAME:    ([[INPUT0:%.+]]: tensor<1x1024x1x3072xf16>,
    // CHECK-SAME:     [[INPUT1:%.+]]: tensor<1x1x1x3072xf16>)
    func.func @MultiplyPipeliningTiling(%arg0: tensor<1x1024x1x3072xf16>, %arg1: tensor<1x1x1x3072xf16>) -> tensor<1x1024x1x3072xf16> {
        %multiply = VPU.Multiply(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>} :
            tensor<1x1024x1x3072xf16>, tensor<1x1x1x3072xf16> -> tensor<1x1024x1x3072xf16>
        return %multiply : tensor<1x1024x1x3072xf16>

    // pipelining tiling is enabled for Multiply operation
    // CHECK:      VPU.Multiply
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 6]
    // CHECK-NOT:       tilingStrategy = [1, 1, 1, 5]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: func.func @InterpSplitOverH
    // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x64x48x80xf16, {order = #NHWC}>
    func.func @InterpSplitOverH(
        %arg0: tensor<1x64x48x80xf16, {order = #NHWC}>)
                -> tensor<1x64x192x320xf16, {order = #NHWC}> {
        %0 = VPU.Interpolate(%arg0) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
            axes_attr = [2, 3],
            operandSegmentSizes = array<i32: 1, 0, 0, 0, 0, 0>,
            sizes_attr = [192, 320],
            tile_offset_attr = [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]} :
            tensor<1x64x48x80xf16, {order = #NHWC}> -> tensor<1x64x192x320xf16, {order = #NHWC}>
        return %0 : tensor<1x64x192x320xf16, {order = #NHWC}>

        // CHECK:  [[INTERP0:%.+]] = VPU.Interpolate([[INPUT]])
        // CHECK-SAME:  tilingStrategy = [1, 1, 1, 6]
        // CHECK-SAME:  : tensor<1x64x48x80xf16, {order = #NHWC}>
        // CHECK-SAME:  -> tensor<1x64x192x320xf16, {order = #NHWC}>

        // CHECK:  return [[INTERP0]] : tensor<1x64x192x320xf16, {order = #NHWC}>
    }

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
  config.Resources 4 of @NCE at 1.850000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
  }

  // CHECK-LABEL: @pipeliningTilingForBigFilter
  func.func @pipeliningTilingForBigFilter(%arg0: tensor<1x1536x1x1xf16, {order = #NHWC}>, %arg1: tensor<8160x1536x1x1x!quant.uniform<i8:f16, 1.000000e+00>, {order = #NHWC}>) -> tensor<1x8160x1x1xf16, {order = #NHWC}> {
    %310 = VPU.NCE.Convolution(%arg0, %arg1) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [8160, 1536, 1, 1], strides = [1, 1]} : tensor<1x1536x1x1xf16, {order = #NHWC}>, tensor<8160x1536x1x1x!quant.uniform<i8:f16, 1.000000e+00>, {order = #NHWC}> -> tensor<1x8160x1x1xf16, {order = #NHWC}>
    return %310 : tensor<1x8160x1x1xf16, {order = #NHWC}>

    // CHECK:         VPU.NCE.Convolution
    // CHECK-SAME:    tilingStrategy = [1, 10, 1, 1]
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL:   @SplitNCEConvOverOH
    // CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x48xf16, {order = #NHWC}>
    func.func @SplitNCEConvOverOH(%arg0: tensor<1x32x64x48xf16, {order = #NHWC}>) -> tensor<1x256x64x48xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

        %0 = VPU.NCE.Convolution(%arg0, %weights) {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [256, 32, 3, 3],
            strides = [1, 1]
        } : tensor<1x32x64x48xf16, {order = #NHWC}>, tensor<256x32x3x3xf16, {order = #NHWC}> -> tensor<1x256x64x48xf16, {order = #NHWC}>

        return %0 : tensor<1x256x64x48xf16, {order = #NHWC}>

        // CHECK-DAG:        [[FILTER:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        // CHECK-SAME:      : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

        // CHECK:        [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER]])
        // CHECK-SAME:          {pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        // CHECK-SAME:          ppe = #VPU.PPEStub<>, rawFilterShape = [256, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 4, 1]}
        // CHECK-SAME:          -> tensor<1x256x64x48xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x256x64x48xf16, {order = #NHWC}>
    }

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i4:f16, 1.3385416666666667>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL:   @SplitI4QuantNCEConvOverOC
    // CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x128x256x4xf16, {order = #NHWC}>
    func.func @SplitI4QuantNCEConvOverOC(%arg0: tensor<1x128x256x4xf16, {order = #NHWC}>) -> tensor<1x6320x256x4xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<6320x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<6320x128x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

        %0 = VPU.NCE.Convolution(%arg0, %weights) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [6320, 128, 1, 1], strides = [1, 1]
        } : tensor<1x128x256x4xf16, {order = #NHWC}>, tensor<6320x128x1x1x!qElemType, {order = #NHWC}> -> tensor<1x6320x256x4xf16, {order = #NHWC}>

        return %0 : tensor<1x6320x256x4xf16, {order = #NHWC}>

        // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<6320x128x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00>
        // CHECK-SAME:      : tensor<6320x128x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]

        // CHECK:           [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]])
        // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
        // CHECK-SAME:          pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        // CHECK-SAME:          rawFilterShape = [6320, 128, 1, 1],
        // CHECK-SAME:          strides = [1, 1],
        // CHECK-SAME:          tilingStrategy = [1, 1, 7, 1]}
        // CHECK-SAME:          -> tensor<1x6320x256x4xf16, {order = #NHWC}>

        // CHECK:           return [[CONV]] : tensor<1x6320x256x4xf16, {order = #NHWC}>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @TileOverCWithBigC
    // CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x1024x4x4xf16, {order = #NHWC}>
    func.func @TileOverCWithBigC(
            %arg0: tensor<1x1024x4x4xf16, {order = #NHWC}>)
                -> tensor<1x8016x4x4xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<8016x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00>
            : tensor<8016x1024x1x1xf16>, [#const.Reorder<#NHWC>]

        %0 = VPU.NCE.Convolution(%arg0, %weights) {
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [8016, 1024, 1, 1],
            strides = [1, 1]
        } : tensor<1x1024x4x4xf16, {order = #NHWC}>, tensor<8016x1024x1x1xf16, {order = #NHWC}> -> tensor<1x8016x4x4xf16, {order = #NHWC}>

        return %0 : tensor<1x8016x4x4xf16, {order = #NHWC}>

    // CHECK-DAG:        [[FILTER:%.+]] = const.Declare tensor<8016x1024x1x1xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<8016x1024x1x1xf16>, [#const.Reorder<#NHWC>]

    // CHECK:        [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER]])
    // CHECK-SAME:          {pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:          ppe = #VPU.PPEStub<>, rawFilterShape = [8016, 1024, 1, 1], strides = [1, 1], tilingStrategy = [1, 32, 1, 1]}
    // CHECK-SAME:          -> tensor<1x8016x4x4xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x8016x4x4xf16, {order = #NHWC}>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitNCEPoolOverH
    // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x340x256xf16, {order = #NHWC}>)
    func.func @SplitNCEPoolOverH(%arg0: tensor<1x16x340x256xf16, {order = #NHWC}>) -> tensor<1x16x340x256xf16, {order = #NHWC}> {
        %0 = VPU.NCE.MaxPool(%arg0) {
            kernel_size = [3, 3],
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            strides = [1, 1]
        } -> tensor<1x16x340x256xf16, {order = #NHWC}>

        return %0 : tensor<1x16x340x256xf16, {order = #NHWC}>

        // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.MaxPool([[INPUT]]) {
        // CHECK-SAME:      pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
        // CHECK-SAME:      tilingStrategy = [1, 1, 8, 1]
        // CHECK-SAME:      } -> tensor<1x16x340x256xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x16x340x256xf16, {order = #NHWC}>
    }

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @TileWithSOK
    // CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x30x30xf16, {order = #NHWC}>
    func.func @TileWithSOK(
            %arg0: tensor<1x32x30x30xf16, {order = #NHWC}>)
                -> tensor<1x768x30x30xf16, {order = #NHWC}> {
        %weights = const.Declare tensor<768x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00>
            : tensor<768x32x7x7xf16>, [#const.Reorder<#NHWC>]

        %0 = VPU.NCE.Convolution(%arg0, %weights) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>,
            pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [768, 32, 7, 7],
            strides = [1, 1]
        } : tensor<1x32x30x30xf16, {order = #NHWC}>, tensor<768x32x7x7xf16, {order = #NHWC}> -> tensor<1x768x30x30xf16, {order = #NHWC}>

        return %0 : tensor<1x768x30x30xf16, {order = #NHWC}>

        // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<768x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00>
        // CHECK-SAME:          : tensor<768x32x7x7xf16>, [#const.Reorder<#NHWC>]

        // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]])
        // CHECK-SAME:          multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverKernel>
        // CHECK-SAME:          pad = #VPU.Padding<left = 3 : i64, right = 3 : i64, top = 3 : i64, bottom = 3 : i64>
        // CHECK-SAME:          rawFilterShape = [768, 32, 7, 7],
        // CHECK-SAME:          strides = [1, 1],
        // CHECK-SAME:          tilingStrategy = [1, 1, 4, 1]}
        // CHECK-SAME:        -> tensor<1x768x30x30xf16, {order = #NHWC}>

        // CHECK:       return [[CONV1]] : tensor<1x768x30x30xf16, {order = #NHWC}>
    }

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
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
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [160, 32, 3, 3],
            strides = [1, 1]
        } : tensor<1x32x80x60xf16, {order = #NHWC}>, !VPU.SparseTensor<data=tensor<160x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<160x1x1x384xi1>, is_weights> -> tensor<1x160x80x60xf16, {order = #NHWC}>

        return %0 : tensor<1x160x80x60xf16, {order = #NHWC}>

        // CHECK-DAG:        [[WEIGHTS:%.+]] = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]

        // CHECK-DAG:        [[WEIGHTS_SM:%.+]] = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00>
        // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

        // CHECK:        [[WEIGHTS_SPARSE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS]], [[WEIGHTS_SM]]) {is_weights} -> !VPU.SparseTensor
        // CHECK-SAME:       data=tensor<160x32x3x3xf16, {order = #NHWC}>,
        // CHECK-SAME:       sparsity_map=tensor<160x1x1x384xi1>, is_weights

        // CHECK:        [[OUTPUT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS_SPARSE]])
        // CHECK-SAME:          pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>
        // CHECK-SAME:          rawFilterShape = [160, 32, 3, 3]
        // CHECK-SAME:          tilingStrategy = [1, 1, 4, 1]
        // CHECK-SAME:          -> tensor<1x160x80x60xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x160x80x60xf16, {order = #NHWC}>
    }

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL: @SplitNCEAveragePoolOverW
    // CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x7x8640xf16, {order = #NHWC}>
    func.func @SplitNCEAveragePoolOverW(%arg0: tensor<1x16x7x8640xf16, {order = #NHWC}>) -> tensor<1x16x1x8640xf16, {order = #NHWC}> {
        %0 = VPU.NCE.AveragePool(%arg0) {kernel_size = [7, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEStub<>, strides = [1, 1]} -> tensor<1x16x1x8640xf16, {order = #NHWC}>
        return %0 : tensor<1x16x1x8640xf16, {order = #NHWC}>

        // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {kernel_size = [7, 1]
        // CHECK-SAME:      tilingStrategy = [1, 1, 1, 5]
        // CHECK-SAME:      -> tensor<1x16x1x8640xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x16x1x8640xf16, {order = #NHWC}>
    }

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @executors {
    config.Resources 6 of @NCE at 1.700000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    }

    // CHECK-LABEL:   @SplitNCECompressConv
    func.func @SplitNCECompressConv(
            %arg0: tensor<1x4x512x512xf16, {order = #NHWC}>,
            %arg1: tensor<64x1x1x160xf16, {order = #NHWC}>,
            %arg2: tensor<64x1x1x4xsi32>)
            -> tensor<1x64x256x256xf16, {order = #NHWC}> {
        %0 = VPU.NCE.CompressConvolution(%arg0, %arg1, %arg2) {
            cm_sp_pattern = 15 : i64,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
            pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [64, 4, 7, 7], strides = [2, 2]
        } -> tensor<1x64x256x256xf16, {order = #NHWC}>

        return %0 : tensor<1x64x256x256xf16, {order = #NHWC}>

        // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.CompressConvolution(%arg0, %arg1, %arg2) {
        // CHECK-SAME:      cm_sp_pattern = 15 : i64, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
        // CHECK-SAME:      pad = #VPU.Padding<left = 3 : i64, right = 2 : i64, top = 3 : i64, bottom = 2 : i64>,
        // CHECK-SAME:      rawFilterShape = [64, 4, 7, 7], strides = [2, 2], tilingStrategy = [1, 1, 4, 1]}
        // CHECK-SAME:      -> tensor<1x64x256x256xf16, {order = #NHWC}>

        // CHECK:       return [[OUTPUT]] : tensor<1x64x256x256xf16, {order = #NHWC}>
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

    // CHECK-LABEL: func.func @PrefetchTilingWithSOHParentConsidered
    // CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x512x14x14x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:        [[WEIGHTS1:%arg[0-9]]]: tensor<512x512x3x3x!qElemType, {order = #NHWC}>,
    // CHECK-SAME:        [[WEIGHTS2:%arg[0-9]]]: tensor<2048x512x1x1x!qElemType, {order = #NHWC}>
    func.func @PrefetchTilingWithSOHParentConsidered(
            %input: tensor<1x512x14x14x!qElemType, {order = #NHWC}>,
            %weights1: tensor<512x512x3x3x!qElemType, {order = #NHWC}>,
            %weights2: tensor<2048x512x1x1x!qElemType, {order = #NHWC}>)
                -> tensor<1x2048x7x7x!qElemType, {order = #NHWC}> {
        %0 = VPU.NCE.Convolution(%input, %weights1) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [512, 512, 3, 3], strides = [2, 2]}
                : tensor<1x512x14x14x!qElemType, {order = #NHWC}>, tensor<512x512x3x3x!qElemType, {order = #NHWC}> -> tensor<1x512x7x7x!qElemType, {order = #NHWC}>
        %1 = VPU.NCE.Convolution(%0, %weights2) {
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
            rawFilterShape = [2048, 512, 1, 1], strides = [1, 1]}
                : tensor<1x512x7x7x!qElemType, {order = #NHWC}>, tensor<2048x512x1x1x!qElemType, {order = #NHWC}> -> tensor<1x2048x7x7x!qElemType, {order = #NHWC}>
        return %1 : tensor<1x2048x7x7x!qElemType, {order = #NHWC}>

        // Prefetching mode is triggered for the child conv
        // with tiled parent memory considered
        // CHECK:       [[PARENT:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]])
        // CHECK-SAME:  tilingStrategy = [1, 8, 1, 1]
        // CHECK:       [[CHILD:%.+]] = VPU.NCE.Convolution([[PARENT]], [[WEIGHTS2]])
        // CHECK-SAME:  tilingStrategy = [1, 2, 1, 1]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @module {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @NCEReduceSumAssignedSOH
func.func @NCEReduceSumAssignedSOH(%arg0: tensor<1x1024x256x256xf16, {order = #NHWC}>) -> tensor<1x1x256x256xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Reduce(%arg0) {
    axes = [1], input_padding = [0, 0, 0, 0], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.reduce_type<SUM>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x1x256x256xf16, {order = #NHWC}>
  return %0 : tensor<1x1x256x256xf16, {order = #NHWC}>
}

// CHECK:      VPU.NCE.Reduce
// CHECK-SAME:     tilingStrategy = [1, 1, 43, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @module {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @NCEReduceSumAssignedSOW
func.func @NCEReduceSumAssignedSOW(%arg0: tensor<1x1024x256x256xf16, {order = #NHWC}>) -> tensor<1x1x256x256xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Reduce(%arg0) {
    axes = [1], input_padding = [0, 0, 0, 0], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>, op_type = #VPU.reduce_type<SUM>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x1x256x256xf16, {order = #NHWC}>
  return %0 : tensor<1x1x256x256xf16, {order = #NHWC}>
}

// CHECK:      VPU.NCE.Reduce
// CHECK-SAME:     tilingStrategy = [1, 1, 128, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @module {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @NCEReduceSumSingleCluster
func.func @NCEReduceSumSingleCluster(%arg0: tensor<1x1024x256x256xf16, {order = #NHWC}>) -> tensor<1x1x256x256xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Reduce(%arg0) {
    axes = [1], input_padding = [0, 0, 0, 0], op_type = #VPU.reduce_type<SUM>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x1x256x256xf16, {order = #NHWC}>
  return %0 : tensor<1x1x256x256xf16, {order = #NHWC}>
}

// CHECK:      VPU.NCE.Reduce
// CHECK-SAME:     tilingStrategy = [1, 1, 52, 2]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @module {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @NCEReduceMeanAssignedSOH
func.func @NCEReduceMeanAssignedSOH(%arg0: tensor<1x1024x256x256xf16, {order = #NHWC}>) -> tensor<1x1x256x256xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Reduce(%arg0) {
    axes = [1], input_padding = [0, 0, 0, 0], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.reduce_type<MEAN>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x1x256x256xf16, {order = #NHWC}>
  return %0 : tensor<1x1x256x256xf16, {order = #NHWC}>
}

// CHECK:      VPU.NCE.Reduce
// CHECK-SAME:     tilingStrategy = [1, 1, 43, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @module {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @NCEReduceMeanAssignedSOW
func.func @NCEReduceMeanAssignedSOW(%arg0: tensor<1x1024x256x256xf16, {order = #NHWC}>) -> tensor<1x1x256x256xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Reduce(%arg0) {
    axes = [1], input_padding = [0, 0, 0, 0], multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverWidth>, op_type = #VPU.reduce_type<MEAN>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x1x256x256xf16, {order = #NHWC}>
  return %0 : tensor<1x1x256x256xf16, {order = #NHWC}>
}

// CHECK:      VPU.NCE.Reduce
// CHECK-SAME:     tilingStrategy = [1, 1, 128, 1]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @module {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}

// CHECK-LABEL: @NCEReduceMeanSingleCluster
func.func @NCEReduceMeanSingleCluster(%arg0: tensor<1x1024x256x256xf16, {order = #NHWC}>) -> tensor<1x1x256x256xf16, {order = #NHWC}> {
  %0 = VPU.NCE.Reduce(%arg0) {
    axes = [1], input_padding = [0, 0, 0, 0], op_type = #VPU.reduce_type<MEAN>,
    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64,
                     scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    } -> tensor<1x1x256x256xf16, {order = #NHWC}>
  return %0 : tensor<1x1x256x256xf16, {order = #NHWC}>
}

// CHECK:      VPU.NCE.Reduce
// CHECK-SAME:     tilingStrategy = [1, 1, 52, 2]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvODUAutopad
module @DepthConvODUAutopad {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}
func.func @main(%input: tensor<1x16x368x368xf16, {order = #NHWC}>) -> tensor<1x8x368x368xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<8x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<8x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %nce = VPU.NCE.DepthConvolution(%input, %weights) {
        input_padding = [0, 8, 0, 0],
        pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
        ppe = #VPU.PPEStub<>,
        rawFilterShape = [8, 1, 3, 3],
        strides = [1, 1]
    } -> tensor<1x8x368x368xf16, {order = #NHWC}>
    return %nce : tensor<1x8x368x368xf16, {order = #NHWC}>

    // Note: the channel dimension should not be tiled, since ODU autopad is used
    // CHECK:       VPU.NCE.DepthConvolution
    // CHECK-SAME:    tilingStrategy = [1, 1, 12, 1]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolODUAutopad
module @MaxPoolODUAutopad {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}
func.func @main(%input: tensor<1x16x368x368xf16, {order = #NHWC}>) -> tensor<1x8x368x368xf16, {order = #NHWC}> {
    %nce = VPU.NCE.MaxPool(%input) {
        input_padding = [0, 8, 0, 0],
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } -> tensor<1x8x368x368xf16, {order = #NHWC}>
    return %nce : tensor<1x8x368x368xf16, {order = #NHWC}>

    // Note: the channel dimension should not be tiled, since ODU autopad is used
    // CHECK:       VPU.NCE.MaxPool
    // CHECK-SAME:    tilingStrategy = [1, 1, 12, 1]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AveragePoolODUAutopad
module @AveragePoolODUAutopad {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}
func.func @main(%input: tensor<1x16x368x368xf16, {order = #NHWC}>) -> tensor<1x8x368x368xf16, {order = #NHWC}> {
    %nce = VPU.NCE.AveragePool(%input) {
        input_padding = [0, 8, 0, 0],
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } -> tensor<1x8x368x368xf16, {order = #NHWC}>
    return %nce : tensor<1x8x368x368xf16, {order = #NHWC}>

    // Note: the channel dimension should not be tiled, since ODU autopad is used
    // CHECK:       VPU.NCE.AveragePool
    // CHECK-SAME:    tilingStrategy = [1, 1, 12, 1]
}
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseODUAutopad
module @EltwiseODUAutopad {
config.Resources 3 of @NCE at 6.000000e+02 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
}
config.PipelineOptions @Options {
    config.Option @config.AutoPaddingODU : true
}
func.func @main(%input: tensor<1x16x368x368xf16, {order = #NHWC}>) -> tensor<1x8x368x368xf16, {order = #NHWC}> {
    %nce = VPU.NCE.Eltwise(%input, %input) {
        input_padding = [0, 8, 0, 0],
        op_type = #VPU.eltwise_type<ADD>,
        ppe = #VPU.PPEStub<>
    } -> tensor<1x8x368x368xf16, {order = #NHWC}>
    return %nce : tensor<1x8x368x368xf16, {order = #NHWC}>

    // Note: the channel dimension should not be tiled, since ODU autopad is used
    // CHECK:       VPU.NCE.Eltwise
    // CHECK-SAME:    tilingStrategy = [1, 1, 8, 1]
}
}
