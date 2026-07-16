//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --scf-vertical-fusion="vf-merge-configuration=GREEDY" --resolve-shaped-type-result-dims --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
    config.Resources 3 of @NCE at 2.100000e+03 MHz {
        config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
        config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
        config.ExecutorResource 2 of @SHAVE_ACT
        config.ExecutorResource 1 of @DPU
    }

    // CHECK-LABEL: @ChainedSkipConnectionBiggestUserIsAlsoSkipSource
    // CHECK-SAME:      ([[ARG0:%arg[0-9]+]]: tensor<1x32x?x?xf32
    func.func @ChainedSkipConnectionBiggestUserIsAlsoSkipSource(%arg0: tensor<1x32x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 32, 1440, 2560]> : tensor<4xsi64>, order = #NCHW}>)
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1440, 2560]> : tensor<4xsi64>, order = #NHWC}> {
        %dwWeights = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> =
            dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Broadcast<0 : i64, 32 : i64>, #const.Reshape<[32, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
        %convWeights = const.Declare tensor<128x128x3x3xf16, {order = #NHWC}> =
            dense<0> : tensor<128x128x3x3xsi8>, [#const.CastElemType<f16>, #const.CastElemType<!quant.uniform<i8:f16, -513.75686274509803:-1>>, #const.ConvertElemType<!quant.uniform<u8:f16, -513.75686274509803:127>>, #const.Dequantize, #const.Reorder<#NHWC>]
        %sparseWeightsData = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> =
            dense<0.0> : tensor<128x32x3x3xf16, {order = #NHWC}>, [#const.Sparsify<false>]
        %sparseWeightsMap = const.Declare tensor<128x1x1x384xi1> =
            dense<0.0> : tensor<128x32x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]
        %sparseWeights = VPU.GroupSparseTensor(%sparseWeightsData, %sparseWeightsMap) {
                is_weights,
                sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<128xi64>, alignment = 16 : i64>}
            -> !VPU.SparseTensor<data=tensor<128x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<128x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<128xi64>, alignment = 16 : i64>>

        %converted = VPU.Convert(%arg0) {
                dstElemType = f16,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
                tilingStrategy = [1, 1, 2, 2560]}
            : tensor<1x32x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 32, 1440, 2560]> : tensor<4xsi64>, order = #NCHW}>
                -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1440, 2560]> : tensor<4xsi64>, order = #NCHW}>
        %permuted = VPU.NCE.Permute(%converted) {
                dstElemType = f16,
                dstOrder = #NHWC,
                expandedChannels = 32 : i64,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeightOverlapped>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                                 clamp_high = 3.4028234663852886E+38 : f64, scale = 5.000000e-01 : f64,
                                 prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
                tilingStrategy = [1, 1, 6, 3]}
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1440, 2560]> : tensor<4xsi64>, order = #NHWC}>
        %depthConv = VPU.NCE.DepthConvolution(%permuted, %dwWeights) rawFilterShape [32, 1, 1, 1] {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                                 clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00],
                                 adder = 0.000000e+00 : f64>,
                
                strides = [1, 1],
                tilingStrategy = [1, 1, 6, 3]}
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1440, 2560]> : tensor<4xsi64>, order = #NHWC}>
        %stridedConv = VPU.NCE.Convolution(%depthConv, %sparseWeights) rawFilterShape [128, 32, 3, 3] {
                resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                                 clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
                                 prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
                
                strides = [2, 2],
                tilingStrategy = [1, 1, 6, 3]}
            : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1440, 2560]> : tensor<4xsi64>, order = #NHWC}>,
              !VPU.SparseTensor<data=tensor<128x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<128x1x1x384xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<0> : tensor<128xi64>, alignment = 16 : i64>>
                -> tensor<1x128x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
        %conv = VPU.NCE.Convolution(%stridedConv, %convWeights) rawFilterShape [128, 128, 3, 3] {
                resultSegmentSizes = array<i32: 1, 0, 0, 0>,
                mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                                 clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
                                 prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
                
                strides = [1, 1],
                tilingStrategy = [1, 1, 8, 3]}
            : tensor<1x128x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>,
              tensor<128x128x3x3xf16, {order = #NHWC}>
                -> tensor<1x128x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
        %depthToSpace = VPU.DepthToSpace(%conv) {
                block_size = 2 : i64,
                mode = #IE.depth_to_space_mode<DEPTH_FIRST>,
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                tilingStrategy = [1, 1, 9, 2]}
            : tensor<1x128x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 128, 720, 1280]> : tensor<4xsi64>, order = #NHWC}>
                -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1440, 2560]> : tensor<4xsi64>, order = #NHWC}>
        %eltwise0 = VPU.NCE.Eltwise(%depthToSpace, %depthConv) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                op_type = #VPU.eltwise_type<ADD>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                                 clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
                                 prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
                tilingStrategy = [1, 1, 11, 3]}
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1440, 2560]> : tensor<4xsi64>, order = #NHWC}>
        %eltwise1 = VPU.NCE.Eltwise(%eltwise0, %permuted) {
                multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
                op_type = #VPU.eltwise_type<ADD>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64,
                                 clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64,
                                 prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>,
                tilingStrategy = [1, 1, 13, 4]}
            -> tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1440, 2560]> : tensor<4xsi64>, order = #NHWC}>

        return %eltwise1 : tensor<1x32x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 1440, 2560]> : tensor<4xsi64>, order = #NHWC}>

        // CHECK:       [[CONVERTED:%.*]] = VPU.Convert([[ARG0]]
        // CHECK:       [[PERMUTED:%.*]] = VPU.NCE.Permute([[CONVERTED]]
        // CHECK:       [[RESULT:%.*]] = scf.for
        // CHECK:         scf.for
        // CHECK:           [[PERMUTED_SLICE:%.*]] = tensor.extract_slice [[PERMUTED]]
        // CHECK:           [[DEPTH_CONV:%.*]] = VPU.NCE.DepthConvolution([[PERMUTED_SLICE]]
        // CHECK:           [[PADDED:%.*]] = tensor.pad [[DEPTH_CONV]]
        // CHECK:           [[STRIDED_CONV:%.*]] = VPU.NCE.Convolution([[PADDED]]
        // CHECK:           [[PADDED_1:%.*]] = tensor.pad [[STRIDED_CONV]]
        // CHECK:           [[CONV:%.*]] = VPU.NCE.Convolution([[PADDED_1]]
        // CHECK:           [[DEPTH_TO_SPACE:%.*]] = VPU.DepthToSpace([[CONV]]
        // CHECK:           [[DEPTH_CONV_SKIP:%.*]] = tensor.extract_slice [[DEPTH_CONV]]
        // CHECK-SAME:          {skip_connection_slice}
        // CHECK:           [[ELTWISE0:%.*]] = VPU.NCE.Eltwise([[DEPTH_TO_SPACE]], [[DEPTH_CONV_SKIP]])
        // CHECK:           [[PERMUTED_SKIP:%.*]] = tensor.extract_slice [[PERMUTED]]
        // CHECK:           [[ELTWISE1:%.*]] = VPU.NCE.Eltwise([[ELTWISE0]], [[PERMUTED_SKIP]])
        // CHECK:           tensor.insert_slice [[ELTWISE1]]
        // CHECK:         scf.yield
        // CHECK:       scf.yield
        // CHECK:       return [[RESULT]] : tensor<1x32x?x?xf16
    }
}
