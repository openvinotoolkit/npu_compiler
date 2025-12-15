//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-vpu="vf-outlining-tile-threshold=1 vf-outlining-instance-threshold=2" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Convolution
module @Convolution attributes {config.arch = #config.arch_kind<NPU37XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
    func.func @main(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16> {
        %cst = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
        %cst_0 = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}>
                      = dense<1.000000e+00> : tensor<48x3x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]

        %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        %1 = VPU.NCE.Permute(%0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>} -> tensor<1x16x62x64xf16, {order = #NHWC}>
        %2 = VPU.Slice %1 [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
        %3 = VPU.NCE.Convolution(%2, %cst_0, %cst) {
              ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [48, 16, 3, 3], strides = [1, 1]}
                  : tensor<1x16x62x62xf16, {order = #NHWC}>, tensor<48x16x3x3xf16, {order = #NHWC}>, tensor<48x1x1x4xsi32> -> tensor<1x48x60x60xf16>
        return %3 : tensor<1x48x60x60xf16>

        // CHECK:       [[CST0:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense_resource<__elided__> : tensor<48x1x1x4xsi32>
        // CHECK:       [[CST1:%.+]] = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense_resource<__elided__> : tensor<48x16x3x3xf16, {order = #NHWC}>, [#const.Sparsify<false>]
        // CHECK:       [[CST2:%.+]] = const.Declare tensor<48x1x1x256xi1> = dense_resource<__elided__> : tensor<48x16x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]

        // CHECK:       [[SPARSE:%.+]] = VPU.GroupSparseTensor([[CST1]], [[CST2]])
        // CHECK-SAME:        {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>} ->
        // CHECK-SAME:        !VPU.SparseTensor<data=tensor<48x16x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<48x1x1x256xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>>

        // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        // CHECK:       [[COPY0:%.+]] = VPU.Copy([[EXPAND]]
        // CHECK-SAME:      -> !VPU.DistributedTensor<1x3x62x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

        // CHECK:       [[PERM:%.+]] = VPU.NCE.Permute([[COPY0]])
        // CHECK-SAME:       -> !VPU.DistributedTensor<1x16x62x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

        // CHECK:       [[COPY1:%.+]] = VPU.Copy([[PERM]]
        // CHECK-SAME:      -> tensor<1x16x62x64xf16, {order = #NHWC}>

        // CHECK:       [[SLICE:%.+]] = VPU.Slice [[COPY1]] [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
        // CHECK:       [[IN:%.+]] = VPU.Copy([[SLICE]]
        // CHECK-SAME:       -> !VPU.DistributedTensor<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

        // CHECK:       [[COPY2:%.+]] = VPU.Copy([[SPARSE]]
        // CHECK-SAME:       -> !VPU.SparseTensor<data=!VPU.DistributedTensor<48x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
        // CHECK-SAME:       sparsity_map=!VPU.DistributedTensor<48x1x1x256xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>>

        // CHECK:       [[COPY3:%.+]] = VPU.Copy([[CST0]]
        // CHECK-SAME:       -> !VPU.DistributedTensor<48x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

        // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[IN]], [[COPY2]], [[COPY3]])
        // CHECK-SAME:          -> !VPU.DistributedTensor<1x48x60x60xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

        // CHECK:       [[OUT:%.+]] = VPU.Copy([[CONV]]
        // CHECK-SAME:          -> tensor<1x48x60x60xf16>

        // CHECK:       return [[OUT]] : tensor<1x48x60x60xf16>
    }
}

// -----

// CHECK-LABEL: @SoftMax
module @SoftMax {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x1000xf16>
    } outputsInfo : {
        DataInfo "softmax" : tensor<1x1000xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1000xf16>) -> tensor<1x1000xf16>
    func.func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
        %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 1000]} : tensor<1x1000xf16> -> tensor<1x1x1x1000xf16>
        %1 = VPU.SoftMax(%0) {axisInd = 3 : i64} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
        %2 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 1000]} : tensor<1x1x1x1000xf16> -> tensor<1x1000xf16>
        return %2 : tensor<1x1000xf16>

        // CHECK:               [[RESHAPE:%.+]] = VPU.AffineReshape([[ARG0]])
        // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 1000]} : tensor<1x1000xf16> -> tensor<1x1x1x1000xf16>
        // CHECK:               [[COPY0:%.+]] = VPU.Copy([[RESHAPE]]
        // CHECK-SAME:              -> !VPU.DistributedTensor<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

        // CHECK:               [[SOFTMAX:%.+]] = VPU.SoftMax([[COPY0]]
        // CHECK-SAME:              -> !VPU.DistributedTensor<1x1x1x1000xf16, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

        // CHECK:               [[COPY1:%.+]] = VPU.Copy([[SOFTMAX]]
        // CHECK-SAME:              -> tensor<1x1x1x1000xf16>

        // CHECK:               [[OUT:%.+]] = VPU.AffineReshape([[COPY1]])
        // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 1000]} : tensor<1x1x1x1000xf16> -> tensor<1x1000xf16>
        // CHECK: return [[OUT]] : tensor<1x1000xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TwoFunctions
module @TwoFunctions {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xui8>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @foo1([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
    func.func @foo1(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16> {
        %cst = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
        %cst_0 = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x3x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]
        %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        %1 = VPU.NCE.Permute(%0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>} -> tensor<1x16x62x64xf16, {order = #NHWC}>
        %2 = VPU.Slice %1 [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
        %3 = VPU.NCE.Convolution(%2, %cst_0, %cst) {
                ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [48, 16, 3, 3], strides = [1, 1]}
                    : tensor<1x16x62x62xf16, {order = #NHWC}>, tensor<48x16x3x3xf16, {order = #NHWC}>, tensor<48x1x1x4xsi32> -> tensor<1x48x60x60xf16>
        return %3 : tensor<1x48x60x60xf16>

        // CHECK:       [[CST0:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense_resource<__elided__> : tensor<48x1x1x4xsi32>
        // CHECK:       [[CST1:%.+]] = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense_resource<__elided__> : tensor<48x16x3x3xf16, {order = #NHWC}>, [#const.Sparsify<false>]
        // CHECK:       [[CST2:%.+]] = const.Declare tensor<48x1x1x256xi1> = dense_resource<__elided__> : tensor<48x16x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]

        // CHECK:       [[SPARSE:%.+]] = VPU.GroupSparseTensor([[CST1]], [[CST2]])
        // CHECK-SAME:        {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>} ->
        // CHECK-SAME:        !VPU.SparseTensor<data=tensor<48x16x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<48x1x1x256xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>>

        // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>

        // CHECK:       [[COPY0:%.+]] = VPU.Copy([[EXPAND]]
        // CHECK-SAME:       -> !VPU.DistributedTensor<1x3x62x64xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

        // CHECK:       [[PERM:%.+]] = VPU.NCE.Permute([[COPY0]]
        // CHECK-SAME:       -> !VPU.DistributedTensor<1x16x62x64xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

        // CHECK:       [[COPY1:%.+]] = VPU.Copy([[PERM]]
        // CHECK-SAME:      -> tensor<1x16x62x64xf16, {order = #NHWC}>

        // CHECK:       [[SLICE:%.+]] = VPU.Slice [[COPY1]] [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
        // CHECK:       [[IN:%.+]] = VPU.Copy([[SLICE]]
        // CHECK-SAME:       -> !VPU.DistributedTensor<1x16x62x62xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>

        // CHECK:       [[COPY2:%.+]] = VPU.Copy([[SPARSE]]
        // CHECK-SAME:       -> !VPU.SparseTensor<data=!VPU.DistributedTensor<48x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>,
        // CHECK-SAME:       sparsity_map=!VPU.DistributedTensor<48x1x1x256xi1, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>>

        // CHECK:       [[COPY3:%.+]] = VPU.Copy([[CST0]]
        // CHECK-SAME:       -> !VPU.DistributedTensor<48x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

        // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[IN]], [[COPY2]], [[COPY3]])
        // CHECK-SAME:       -> !VPU.DistributedTensor<1x48x60x60xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

        // CHECK:       [[OUT:%.+]] = VPU.Copy([[CONV]]
        // CHECK-SAME:          -> tensor<1x48x60x60xf16>

        // CHECK:       return [[OUT]] : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @foo2([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    func.func @foo2(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        return %0 : tensor<1x48x60x60xf16>

        // CHECK:       [[COPY:%.+]] = VPU.Copy([[ARG0]]
        // CHECK-SAME:       -> !VPU.DistributedTensor<1x48x60x60xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

        // CHECK:       [[SOFTMAX:%.+]] = VPU.SoftMax([[COPY]]
        // CHECK-SAME:       -> !VPU.DistributedTensor<1x48x60x60xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

        // CHECK:       [[OUT:%.+]] = VPU.Copy([[SOFTMAX]]
        // CHECK-SAME:          -> tensor<1x48x60x60xf16>

        // CHECK: return [[OUT]] : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func private @main_outline1([[ARG0:%.+]]: tensor<1x3x62x62xui8>) -> tensor<1x3x62x62xf16>
    // CHECK:       [[COPY:%.+]] = VPU.Copy([[ARG0]]
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x3x62x62xui8, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[CONVERT:%.+]] = VPU.Convert([[COPY]]
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x3x62x62xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

    // CHECK:       [[COPY_BACK:%.+]] = VPU.Copy([[CONVERT]]
    // CHECK-SAME:          -> tensor<1x3x62x62xf16>

    // CHECK: return [[COPY_BACK]] : tensor<1x3x62x62xf16>


    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xui8>) -> tensor<1x48x60x60xf16>
    func.func @main(%arg0: tensor<1x3x62x62xui8>) -> tensor<1x48x60x60xf16> {
        %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x3x62x62xui8> -> tensor<1x3x62x62xf16>
        %1 = call @foo1(%0) : (tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
        %2 = call @foo2(%1) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        return %2 : tensor<1x48x60x60xf16>

        // CHECK:       [[OUTLINE1_RES:%.+]] = call @main_outline1([[ARG0]]) : (tensor<1x3x62x62xui8>) -> tensor<1x3x62x62xf16>
        // CHECK:       [[FOO1_RES:%.+]] = call @foo1([[OUTLINE1_RES]]) : (tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
        // CHECK:       [[FOO2_RES:%.+]] = call @foo2([[FOO1_RES]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        // CHECK:       return [[FOO2_RES]] : tensor<1x48x60x60xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @RepeatingBlocks
module @RepeatingBlocks {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x48x60x60xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func private @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
    func.func private @main_fn1(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %shape_cast1 = VPU.ShapeCast {shape = [1, 48, 225, 16]} inputs(%arg0 : tensor<1x48x60x60xf16>) -> tensor<1x48x225x16xf16>
        %permute = VPU.NCE.Permute(%shape_cast1) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 48 : i64, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64>} -> tensor<1x48x225x16xf16, {order = #NHWC}>
        %shape_cast2 = VPU.ShapeCast {shape = [1, 48, 60, 60]} inputs(%permute : tensor<1x48x225x16xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>

        %cst_weights_table = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
        %cst_weights = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x48x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
        %conv = VPU.NCE.Convolution(%shape_cast2, %cst_weights, %cst_weights_table) {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
            rawFilterShape = [48, 48, 3, 3], strides = [1, 1]
        } : tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<48x48x3x3xf16, {order = #NHWC}>, tensor<48x1x1x4xsi32> -> tensor<1x48x60x60xf16>

        return %conv : tensor<1x48x60x60xf16>

        // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x48x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
        // CHECK-DAG:   [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>

        // CHECK:       [[SHAPE_CAST1:%.+]] = VPU.ShapeCast {shape = [1, 48, 225, 16]} inputs([[ARG0]] : tensor<1x48x60x60xf16>) -> tensor<1x48x225x16xf16>
        // CHECK:       [[INPUT_COPY1:%.+]] = VPU.Copy([[SHAPE_CAST1]]
        // CHECK-SAME:           -> !VPU.DistributedTensor<1x48x225x16xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

        // CHECK:       [[PERM:%.+]] = VPU.NCE.Permute([[INPUT_COPY1]]
        // CHECK-SAME:           -> !VPU.DistributedTensor<1x48x225x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

        // CHECK:       [[INPUT_COPY2:%.+]] = VPU.Copy([[PERM]]
        // CHECK-SAME:           -> tensor<1x48x225x16xf16, {order = #NHWC}>

        // CHECK:       [[SHAPE_CAST2:%.+]] = VPU.ShapeCast {shape = [1, 48, 60, 60]} inputs([[INPUT_COPY2]] : tensor<1x48x225x16xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
        // CHECK:       [[INPUT_COPY3:%.+]] = VPU.Copy([[SHAPE_CAST2]]
        // CHECK-SAME:           -> !VPU.DistributedTensor<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

        // CHECK:       [[COPY_WEIGHTS:%.+]] = VPU.Copy([[CST_WEIGHTS]]
        // CHECK-SAME:           -> !VPU.DistributedTensor<48x48x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

        // CHECK:       [[COPY_WEIGHTS_TABLE:%.+]] = VPU.Copy([[CST_WEIGHTS_TABLE]]
        // CHECK-SAME:           -> !VPU.DistributedTensor<48x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

        // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT_COPY3]], [[COPY_WEIGHTS]], [[COPY_WEIGHTS_TABLE]])
        // CHECK-SAME:           -> !VPU.DistributedTensor<1x48x60x60xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

        // CHECK:       [[OUTPUT_COPY:%.+]] = VPU.Copy([[CONV]]
        // CHECK-SAME:           -> tensor<1x48x60x60xf16>

        // CHECK:       return [[OUTPUT_COPY]]
    }

    // CHECK: func.func private @main_outline1([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf16> {
    // CHECK:       [[COPY:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x48x60x60xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:       [[CONVERT:%.+]] = VPU.Convert([[COPY]]
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x48x60x60xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters = 2 : i64}>

    // CHECK:       [[COPY_BACK:%.+]] = VPU.Copy([[CONVERT]]
    // CHECK-SAME:          -> tensor<1x48x60x60xf16>

    // CHECK:       return [[COPY_BACK]]


    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf16> {
    func.func @main(%input: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf16> {
        %convert = VPU.Convert(%input) {dstElemType = f16} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf16>
        %call1 = call @main_fn1(%convert) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        %call2 = call @main_fn1(%call1) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        return %call2 : tensor<1x48x60x60xf16>

        // CHECK:       [[OUTLINE1_RES:%.+]] = call @main_outline1([[INPUT]]) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf16>
        // CHECK:       [[CALL1:%.+]] = call @main_fn1([[OUTLINE1_RES]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        // CHECK:       [[CALL2:%.+]] = call @main_fn1([[CALL1]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        // CHECK:       return [[CALL2]] : tensor<1x48x60x60xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

module @VerticalFusionOutlining {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x3x256x256xf32, {order = #NHWC}>
    } outputsInfo : {
        DataInfo "output" : tensor<1x16x256x256xf16, {order = #NHWC}>
    }

    func.func @main(%arg0: tensor<1x3x256x256xf32, {order = #NHWC}>) -> tensor<1x16x256x256xf16, {order = #NHWC}> {
        %cst = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
        %cst_0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

        %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x3x256x256xf32, {order = #NHWC}> -> tensor<1x3x256x256xf16, {order = #NHWC}>
        %1 = VPU.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x256x256xf16, {order = #NHWC}> -> tensor<1x16x256x256xf16, {order = #NHWC}>

        %2 = VPU.NCE.Convolution(%1, %cst_0, %cst) {
                ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1]}
                    : tensor<1x16x256x256xf16, {order = #NHWC}>, tensor<16x16x3x3xf16, {order = #NHWC}>, tensor<16x1x1x4xsi32> -> tensor<1x16x256x256xf16, {order = #NHWC}>
        %3 = VPU.SoftMax(%2) {axisInd = 3 : i64} : tensor<1x16x256x256xf16, {order = #NHWC}> -> tensor<1x16x256x256xf16, {order = #NHWC}>

        return %3  : tensor<1x16x256x256xf16, {order = #NHWC}>
    }
}

// CHECK:     func.func private @main_vf1([[ARG0:%.+]]: tensor<1x3x256x256xf32, {order = #NHWC}>) -> tensor<1x3x256x256xf16, {order = #NHWC}> {
// CHECK:       [[COPY_0:%.+]] = VPU.Copy([[ARG0]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x3x256x256xf32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

// CHECK:       [[CONVERT:%.+]] = VPU.Convert([[COPY_0]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x3x256x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>

// CHECK:       [[COPY_1:%.+]] = VPU.Copy([[CONVERT]]
// CHECK-SAME:       -> tensor<1x3x256x256xf16, {order = #NHWC}>
// CHECK:       return [[COPY_1]] : tensor<1x3x256x256xf16, {order = #NHWC}>

// CHECK:     func.func private @main_vf2([[ARG0:%.+]]: tensor<1x3x256x256xf16, {order = #NHWC}>) -> tensor<1x16x256x256xf16, {order = #NHWC}> {
// CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
// CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x16x3x3xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
// CHECK-DAG:   [[EXPAND:%.+]] = VPU.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} : tensor<1x3x256x256xf16, {order = #NHWC}> -> tensor<1x16x256x256xf16, {order = #NHWC}>

// CHECK:       [[SLICE_0:%.+]] = VPU.Slice [[EXPAND]] [0, 0, 0, 0] [1, 16, 87, 256] : tensor<1x16x256x256xf16, {order = #NHWC}> to tensor<1x16x87x256xf16, {order = #NHWC}>
// CHECK:       [[COPY0_0:%.+]] = VPU.Copy([[SLICE_0]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x16x87x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:       [[COPY0_1:%.+]] = VPU.Copy([[CST_0]]
// CHECK-SAME:       -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:       [[COPY0_2:%.+]] = VPU.Copy([[CST]]
// CHECK-SAME:       -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:       [[CONV0:%.+]] = VPU.NCE.Convolution([[COPY0_0]], [[COPY0_1]], [[COPY0_2]])
// CHECK-SAME:       -> !VPU.DistributedTensor<1x16x86x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:       [[SOFTMAX0:%.+]] = VPU.SoftMax([[CONV0]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x16x86x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:       [[COPY0_3:%.+]] = VPU.Copy([[SOFTMAX0]]
// CHECK-SAME:       -> tensor<1x16x86x256xf16, {order = #NHWC}>

// CHECK:       [[SLICE_1:%.+]] = VPU.Slice [[EXPAND]] [0, 0, 85, 0] [1, 16, 87, 256] : tensor<1x16x256x256xf16, {order = #NHWC}> to tensor<1x16x87x256xf16, {order = #NHWC}>
// CHECK:       [[COPY1_0:%.+]] = VPU.Copy([[SLICE_1]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x16x87x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:       [[COPY1_1:%.+]] = VPU.Copy([[CST_0]]
// CHECK-SAME:       -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:       [[COPY1_2:%.+]] = VPU.Copy([[CST]]
// CHECK-SAME:       -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[COPY1_0]], [[COPY1_1]], [[COPY1_2]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x16x85x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:       [[SOFTMAX1:%.+]] = VPU.SoftMax([[CONV1]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x16x85x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:       [[COPY1_3:%.+]] = VPU.Copy([[SOFTMAX1]]
// CHECK-SAME:       -> tensor<1x16x85x256xf16, {order = #NHWC}>

// CHECK:       [[SLICE_2:%.+]] = VPU.Slice [[EXPAND]] [0, 0, 170, 0] [1, 16, 86, 256] : tensor<1x16x256x256xf16, {order = #NHWC}> to tensor<1x16x86x256xf16, {order = #NHWC}>
// CHECK:       [[COPY2_0:%.+]] = VPU.Copy([[SLICE_2]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x16x86x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:       [[COPY2_1:%.+]] = VPU.Copy([[CST_0]]
// CHECK-SAME:       -> !VPU.DistributedTensor<16x16x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:       [[COPY2_2:%.+]] = VPU.Copy([[CST]]
// CHECK-SAME:       -> !VPU.DistributedTensor<16x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
// CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[COPY2_0]], [[COPY2_1]], [[COPY2_2]])
// CHECK-SAME:       -> !VPU.DistributedTensor<1x16x85x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:       [[SOFTMAX2:%.+]] = VPU.SoftMax([[CONV2]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x16x85x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
// CHECK:       [[COPY2_3:%.+]] = VPU.Copy([[SOFTMAX2]]
// CHECK-SAME:       -> tensor<1x16x85x256xf16, {order = #NHWC}>

// CHECK:       [[CONCAT:%.+]] = VPU.Concat([[COPY0_3]], [[COPY1_3]], [[COPY2_3]])
// CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 86, 0], [0, 0, 171, 0]]}
// CHECK-SAME:           : tensor<1x16x86x256xf16, {order = #NHWC}>, tensor<1x16x85x256xf16, {order = #NHWC}>, tensor<1x16x85x256xf16, {order = #NHWC}> -> tensor<1x16x256x256xf16, {order = #NHWC}>
// CHECK:       return [[CONCAT]] : tensor<1x16x256x256xf16, {order = #NHWC}>

// CHECK:     func.func @main([[ARG0:%.+]]: tensor<1x3x256x256xf32, {order = #NHWC}>) -> tensor<1x16x256x256xf16, {order = #NHWC}> {
// CHECK:       [[CALL0:%.+]] = call @main_vf1([[ARG0]]) : (tensor<1x3x256x256xf32, {order = #NHWC}>) -> tensor<1x3x256x256xf16, {order = #NHWC}>
// CHECK:       [[CALL1:%.+]] = call @main_vf2([[CALL0]]) : (tensor<1x3x256x256xf16, {order = #NHWC}>) -> tensor<1x16x256x256xf16, {order = #NHWC}>
// CHECK:       return [[CALL1]] : tensor<1x16x256x256xf16, {order = #NHWC}>

// -----


// CHECK-LABEL: @AdjustMemorySpaceAndOptimizeSharedInputCopyForConcat1T
// Check whether OptimizeSharedInputCopyForConcat is applied after reordering with AdjustMemorySpace on 1T (E#156584)
// The pattern that OptimizeSharedInputCopyForConcat expects: Copy(CMX2DDR) -> Concat(DDR) -> {Concat(DDR) -> Slice(DDR) -> Copy(DDR2CMX)} x N, where N >= 2
// On 1 Tile, CMX won't be involved until AdjustMemorySpace, so OptimizeSharedInputCopyForConcat will fail to match due to the missing CopyOps at the beginning and end
// By bringing AdjustMemorySpace in front of OptimizeSharedInputCopyForConcat, the pattern will be matched and rewritten to: Copy(CMX2DDR) -> Concat(DDR) -> {Slice(DDR) -> Copy(DDR2CMX) -> Concat(CMX)}
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @AdjustMemorySpaceAndOptimizeSharedInputCopyForConcat1T {
  config.Resources 1 of @NCE at 1.300000e+03 MHz {
    config.MemoryResource 1784217 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1982464 bytes of @CMX_NN {config.bandwidth = 32 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @SHAVE_NN
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 1 of @DMA_NN
  config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 8 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" tensorNames = ["input"] : tensor<1x3x128x128xf32>
  } outputsInfo : {
    DataInfo "output" friendlyName = "output" : tensor<1x32x128x128xf32>
  }
  func.func @main(%arg0: tensor<1x3x128x128xf32>) -> tensor<1x32x128x128xf32> {
    %cst = const.Declare tensor<1024x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x144x1x3xf16, {order = #NHWC}>, [#const.Reshape<[1024, 48, 3, 3]>]
    %cst_0 = const.Declare tensor<1x1024x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf32>, [#const.CastElemType<f16>, #const.Broadcast<1 : i64, 1024 : i64>, #const.Reshape<[1, 1024, 1, 1]>]
    %cst_1 = const.Declare tensor<32x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x128x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<32x96x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x96x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<32x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x64x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf32>, [#const.CastElemType<f16>]
    %cst_5 = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf32>, [#const.CastElemType<f16>]
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x3x128x128xf32> -> tensor<1x3x128x128xf16>
    %1 = VPU.NCE.Permute(%0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 3 : i64, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>} -> tensor<1x3x128x128xf16, {order = #NHWC}>
    %2 = VPU.ShapeCast {shape = [1, 48, 128, 8]} inputs(%1 : tensor<1x3x128x128xf16, {order = #NHWC}>) -> tensor<1x48x128x8xf16, {order = #NHWC}>
    %cst_6 = const.Declare tensor<1024x1x1x4xsi32> = dense<1> : tensor<1024x1x1x4xsi32>
    %3 = VPU.NCE.Convolution(%2, %cst, %cst_6) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [1024, 48, 3, 3], strides = [1, 1]} : tensor<1x48x128x8xf16, {order = #NHWC}>, tensor<1024x48x3x3xf16, {order = #NHWC}>, tensor<1024x1x1x4xsi32> -> tensor<1x1024x128x8xf16, {order = #NHWC}>
    %4 = VPU.ShapeCast {shape = [1, 64, 128, 128]} inputs(%3 : tensor<1x1024x128x8xf16, {order = #NHWC}>) -> tensor<1x64x128x128xf16, {order = #NHWC}>
    %cst_7 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %5 = VPU.NCE.Convolution(%4, %cst_3, %cst_7) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64, fp_prelu_alpha = 0.199951171875 : f64>, rawFilterShape = [32, 64, 3, 3], strides = [1, 1]} : tensor<1x64x128x128xf16, {order = #NHWC}>, tensor<32x64x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x128x128xf16, {order = #NHWC}>
    %6 = VPU.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x128x128xf16, {order = #NHWC}>, tensor<1x32x128x128xf16, {order = #NHWC}> -> tensor<1x96x128x128xf16, {order = #NHWC}>
    %cst_8 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %7 = VPU.NCE.Convolution(%6, %cst_2, %cst_8) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LPRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1638 : i64, lrelu_shift = 13 : i64, fp_prelu_alpha = 0.199951171875 : f64>, rawFilterShape = [32, 96, 3, 3], strides = [1, 1]} : tensor<1x96x128x128xf16, {order = #NHWC}>, tensor<32x96x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x128x128xf16, {order = #NHWC}>
    %8 = VPU.Concat(%4, %5, %7) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]} : tensor<1x64x128x128xf16, {order = #NHWC}>, tensor<1x32x128x128xf16, {order = #NHWC}>, tensor<1x32x128x128xf16, {order = #NHWC}> -> tensor<1x128x128x128xf16, {order = #NHWC}>
    %cst_9 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %9 = VPU.NCE.Convolution(%8, %cst_1, %cst_9) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>, rawFilterShape = [32, 128, 3, 3], strides = [1, 1]} : tensor<1x128x128x128xf16, {order = #NHWC}>, tensor<32x128x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x128x128xf16>
    %10 = VPU.Convert(%9) {dstElemType = f32} : tensor<1x32x128x128xf16> -> tensor<1x32x128x128xf32>
    return %10 : tensor<1x32x128x128xf32>

    // CHECK:   func.func @main([[ARG0:%.+]]: tensor<1x3x128x128xf32>) -> tensor<1x32x128x128xf32> {
        // CHECK:       [[SHAPECAST1:%.+]] = VPU.ShapeCast {shape = [1, 64, 128, 128]} inputs({{%.+}} : tensor<1x1024x128x8xf16, {order = #NHWC}>) -> tensor<1x64x128x128xf16, {order = #NHWC}>
        // CHECK:       [[CONCAT1:%.+]] = VPU.Concat({{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}) {static_offsets =
        // CHECK-SAME{LITERAL}:  [[0, 0, 0, 0], [0, 0, 0, 26], [0, 0, 0, 52], [0, 0, 0, 78], [0, 0, 0, 103]]} : tensor<1x32x128x26xf16, {order = #NHWC}>, tensor<1x32x128x26xf16, {order = #NHWC}>, tensor<1x32x128x26xf16, {order = #NHWC}>, tensor<1x32x128x25xf16, {order = #NHWC}>, tensor<1x32x128x25xf16, {order = #NHWC}> -> tensor<1x32x128x128xf16, {order = #NHWC}>
        // CHECK:       [[SLICE9:%.+]] = VPU.Slice [[CONCAT1]] [0, 0, 0, 0] [1, 32, 23, 128] : tensor<1x32x128x128xf16, {order = #NHWC}> to tensor<1x32x23x128xf16, {order = #NHWC}>
        // CHECK:       [[COPY38:%.+]] = VPU.Copy([[SLICE9]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x23x128xf16, {order = #NHWC}> -> tensor<1x32x23x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE10:%.+]] = VPU.Slice [[SHAPECAST1]] [0, 0, 0, 0] [1, 64, 23, 128] : tensor<1x64x128x128xf16, {order = #NHWC}> to tensor<1x64x23x128xf16, {order = #NHWC}>
        // CHECK:       [[COPY39:%.+]] = VPU.Copy([[SLICE10]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x64x23x128xf16, {order = #NHWC}> -> tensor<1x64x23x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE11:%.+]] = VPU.Slice [[CONCAT1]] [0, 0, 21, 0] [1, 32, 24, 128] : tensor<1x32x128x128xf16, {order = #NHWC}> to tensor<1x32x24x128xf16, {order = #NHWC}>
        // CHECK:       [[COPY40:%.+]] = VPU.Copy([[SLICE11]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x24x128xf16, {order = #NHWC}> -> tensor<1x32x24x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE12:%.+]] = VPU.Slice [[SHAPECAST1]] [0, 0, 21, 0] [1, 64, 24, 128] : tensor<1x64x128x128xf16, {order = #NHWC}> to tensor<1x64x24x128xf16, {order = #NHWC}>
        // CHECK:       [[COPY41:%.+]] = VPU.Copy([[SLICE12]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x64x24x128xf16, {order = #NHWC}> -> tensor<1x64x24x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE13:%.+]] = VPU.Slice [[CONCAT1]] [0, 0, 43, 0] [1, 32, 23, 128] : tensor<1x32x128x128xf16, {order = #NHWC}> to tensor<1x32x23x128xf16, {order = #NHWC}>
        // CHECK:       [[COPY42:%.+]] = VPU.Copy([[SLICE13]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x23x128xf16, {order = #NHWC}> -> tensor<1x32x23x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE14:%.+]] = VPU.Slice [[SHAPECAST1]] [0, 0, 43, 0] [1, 64, 23, 128] : tensor<1x64x128x128xf16, {order = #NHWC}> to tensor<1x64x23x128xf16, {order = #NHWC}>
        // CHECK:       [[COPY43:%.+]] = VPU.Copy([[SLICE14]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x64x23x128xf16, {order = #NHWC}> -> tensor<1x64x23x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE15:%.+]] = VPU.Slice [[CONCAT1]] [0, 0, 64, 0] [1, 32, 23, 128] : tensor<1x32x128x128xf16, {order = #NHWC}> to tensor<1x32x23x128xf16, {order = #NHWC}>
        // CHECK:       [[COPY44:%.+]] = VPU.Copy([[SLICE15]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x23x128xf16, {order = #NHWC}> -> tensor<1x32x23x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE16:%.+]] = VPU.Slice [[SHAPECAST1]] [0, 0, 64, 0] [1, 64, 23, 128] : tensor<1x64x128x128xf16, {order = #NHWC}> to tensor<1x64x23x128xf16, {order = #NHWC}>
        // CHECK:       [[COPY45:%.+]] = VPU.Copy([[SLICE16]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x64x23x128xf16, {order = #NHWC}> -> tensor<1x64x23x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE17:%.+]] = VPU.Slice [[CONCAT1]] [0, 0, 85, 0] [1, 32, 23, 128] : tensor<1x32x128x128xf16, {order = #NHWC}> to tensor<1x32x23x128xf16, {order = #NHWC}>
        // CHECK:       [[COPY46:%.+]] = VPU.Copy([[SLICE17]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x23x128xf16, {order = #NHWC}> -> tensor<1x32x23x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE18:%.+]] = VPU.Slice [[SHAPECAST1]] [0, 0, 85, 0] [1, 64, 23, 128] : tensor<1x64x128x128xf16, {order = #NHWC}> to tensor<1x64x23x128xf16, {order = #NHWC}>
        // CHECK:       [[COPY47:%.+]] = VPU.Copy([[SLICE18]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x64x23x128xf16, {order = #NHWC}> -> tensor<1x64x23x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE19:%.+]] = VPU.Slice [[CONCAT1]] [0, 0, 106, 0] [1, 32, 22, 128] : tensor<1x32x128x128xf16, {order = #NHWC}> to tensor<1x32x22x128xf16, {order = #NHWC}>
        // CHECK:       [[COPY48:%.+]] = VPU.Copy([[SLICE19]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x22x128xf16, {order = #NHWC}> -> tensor<1x32x22x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE20:%.+]] = VPU.Slice [[SHAPECAST1]] [0, 0, 106, 0] [1, 64, 22, 128] : tensor<1x64x128x128xf16, {order = #NHWC}> to tensor<1x64x22x128xf16, {order = #NHWC}>
        // CHECK:       [[COPY49:%.+]] = VPU.Copy([[SLICE20]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x64x22x128xf16, {order = #NHWC}> -> tensor<1x64x22x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // Optimization result: CMX concatenations (order matches the actual output %87->%85, %83->%81, etc.)
        // CHECK:       [[CONCAT_LAST:%.+]] = VPU.Concat([[COPY49]], [[COPY48]]) {static_offsets =
        // CHECK-SAME{LITERAL}:  [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x22x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x22x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x96x22x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[CONCAT_5TH:%.+]] = VPU.Concat([[COPY47]], [[COPY46]]) {static_offsets =
        // CHECK-SAME{LITERAL}:  [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x23x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x23x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x96x23x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[CONCAT_4TH:%.+]] = VPU.Concat([[COPY45]], [[COPY44]]) {static_offsets =
        // CHECK-SAME{LITERAL}:  [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x23x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x23x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x96x23x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[CONCAT_3RD:%.+]] = VPU.Concat([[COPY43]], [[COPY42]]) {static_offsets =
        // CHECK-SAME{LITERAL}:  [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x23x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x23x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x96x23x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[CONCAT_2ND:%.+]] = VPU.Concat([[COPY41]], [[COPY40]]) {static_offsets =
        // CHECK-SAME{LITERAL}:  [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x24x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x24x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x96x24x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[CONCAT_1ST:%.+]] = VPU.Concat([[COPY39]], [[COPY38]]) {static_offsets =
        // CHECK-SAME{LITERAL}:  [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x23x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x23x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x96x23x128xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

        // CHECK:       return {{%.+}} : tensor<1x32x128x128xf32>
  }
}
