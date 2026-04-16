//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true weights-table-reuse-mode=VF_ENABLED" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-vpu="vf-outlining-tile-threshold=1 vf-outlining-instance-threshold=2" %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Convolution
module @Convolution attributes {config.arch = #config.arch_kind<NPU50XX>, config.compilationMode = #config.compilation_mode<DefaultHW>} {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    config.Resources 1 of @NCE at 1.300000e+03 MHz

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
    func.func @main(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16> {
        %cst = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
        %cst_0 = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}>
                      = dense<1.000000e+00> : tensor<48x3x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]

        %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        %1 = VPU.NCE.Permute(%0) {
            dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
        } -> tensor<1x16x62x64xf16, {order = #NHWC}>
        %2 = VPU.Slice %1 [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
        %3 = VPU.NCE.Convolution(%2, %cst_0, %cst) {
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
              ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
              rawFilterShape = [48, 16, 3, 3], strides = [1, 1]
              } : tensor<1x16x62x62xf16, {order = #NHWC}>, tensor<48x16x3x3xf16, {order = #NHWC}>, tensor<48x1x1x4xsi32> -> tensor<1x48x60x60xf16>
        return %3 : tensor<1x48x60x60xf16>

        // CHECK:       [[CST0:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense_resource<__elided__> : tensor<48x1x1x4xsi32>
        // CHECK:       [[CST1:%.+]] = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense_resource<__elided__> : tensor<48x16x3x3xf16, {order = #NHWC}>, [#const.Sparsify<false>]
        // CHECK:       [[CST2:%.+]] = const.Declare tensor<48x1x1x256xi1> = dense_resource<__elided__> : tensor<48x16x3x3xf16, {order = #NHWC}>, [#const.GetSparsityMap]

        // CHECK:       [[SPARSE:%.+]] = VPU.GroupSparseTensor([[CST1]], [[CST2]])
        // CHECK-SAME:        {is_weights, sparsity_compression = #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>} ->
        // CHECK-SAME:        !VPU.SparseTensor<data=tensor<48x16x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<48x1x1x256xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>>

        // CHECK:       [[EXPAND:%.+]] = VPU.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        // CHECK:       [[COPY0:%.+]] = VPU.Copy([[EXPAND]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x3x62x64xf16> -> tensor<1x3x62x64xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>

        // Track E#164817 - Correct the cost of this NCEPermute
        // CHECK:       [[PERM:%.+]] = VPU.NCE.Permute([[COPY0]])
        // CHECK-SAME:           {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, minimumHardwareExecutionCost = 4294967195 : i64, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>}
        // CHECK-SAME:           -> tensor<1x16x62x64xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> {
        // CHECK:                   VPU.DPU.Workload
        // CHECK-SAME:              inOffsets [0, 0, 0, 0] inSizes [1, 3, 62, 64] outOffsets [0, 0, 0, 0] outSizes [1, 3, 62, 64] pad [0, 0, 0, 0] <CUBOID_16x16>
        // CHECK:                 }

        // CHECK:       [[COPY1:%.+]] = VPU.Copy([[PERM]]) : tensor<1x16x62x64xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x16x62x64xf16, {order = #NHWC}>

        // CHECK:       [[SLICE:%.+]] = VPU.Slice [[COPY1]] [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
        // CHECK:       [[COPY2:%.+]] = VPU.Copy([[SLICE]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x16x62x62xf16, {order = #NHWC}> -> tensor<1x16x62x62xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

        // CHECK:       [[COPY1:%.+]] = VPU.Copy([[SPARSE]])
        // CHECK-SAME:      {out_mem_space = [@CMX_NN, 0]} : !VPU.SparseTensor<data=tensor<48x16x3x3xf16, {order = #NHWC}>,
        // CHECK-SAME:      sparsity_map=tensor<48x1x1x256xi1>, is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>>
        // CHECK-SAME:      -> !VPU.SparseTensor<data=tensor<48x16x3x3xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, sparsity_map=tensor<48x1x1x256xi1, {mem_space = [@CMX_NN, 0], order = #NCHW}>,
        // CHECK-SAME:      is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>>



        // CHECK:       [[COPY3:%.+]] = VPU.Copy([[CST0]]) {out_mem_space = [@CMX_NN, 0]} : tensor<48x1x1x4xsi32> -> tensor<48x1x1x4xsi32, {mem_space = [@CMX_NN, 0], order = #NCHW}>

        // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[COPY2]], [[COPY1]], [[COPY3]]) {minimumHardwareExecutionCost = 42679 : i64,
        // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
        // CHECK-SAME:      rawFilterShape = [48, 16, 3, 3], strides = [1, 1]}
        // CHECK-SAME:      -> tensor<1x48x60x60xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}> {
        // CHECK:               VPU.DPU.Workload
        // CHECK-SAME:          inOffsets [0, 0, 0, 0] inSizes [1, 16, 62, 62] outOffsets [0, 0, 0, 0] outSizes [1, 48, 60, 60] pad [0, 0, 0, 0]
        // CHECK-SAME:          <CUBOID_8x16>
        // CHECK-NEXT:          }


        // CHECK:       [[COPY4:%.+]] = VPU.Copy([[CONV]]) : tensor<1x48x60x60xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}> -> tensor<1x48x60x60xf16>

        // CHECK:       return [[COPY4]] : tensor<1x48x60x60xf16>
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

    config.Resources 3 of @NCE at 6.000000e+02 MHz

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x1000xf16>) -> tensor<1x1000xf16>
    func.func @main(%arg0: tensor<1x1000xf16>) -> tensor<1x1000xf16> {
        %0 = VPU.AffineReshape(%arg0) {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 1000]} : tensor<1x1000xf16> -> tensor<1x1x1x1000xf16>
        %1 = VPU.SoftMax(%0) {axisInd = 3 : i64} : tensor<1x1x1x1000xf16> -> tensor<1x1x1x1000xf16>
        %2 = VPU.AffineReshape(%1) {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 1000]} : tensor<1x1x1x1000xf16> -> tensor<1x1000xf16>
        return %2 : tensor<1x1000xf16>

        // CHECK:               [[RESHAPE:%.+]] = VPU.AffineReshape([[ARG0]])
        // CHECK-SAME{LITERAL}:     {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 1000]} : tensor<1x1000xf16> -> tensor<1x1x1x1000xf16>
        // CHECK:               [[COPY0:%.+]] = VPU.Copy([[RESHAPE]]
        // CHECK-SAME:              -> !VPU.DistributedTensor<1x1x1x1000xf16, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

        // CHECK:               [[SOFTMAX:%.+]] = VPU.SoftMax([[COPY0]]
        // CHECK-SAME:              -> !VPU.DistributedTensor<1x1x1x1000xf16, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 1, 1, 1000], [1, 1, 1, 1000], [1, 1, 1, 1000]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

        // CHECK:               [[COPY1:%.+]] = VPU.Copy([[SOFTMAX]]
        // CHECK-SAME:              -> tensor<1x1x1x1000xf16>

        // CHECK:               [[OUT:%.+]] = VPU.AffineReshape([[COPY1]])
        // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 1000]} : tensor<1x1x1x1000xf16> -> tensor<1x1000xf16>
        // CHECK:               return [[OUT]] : tensor<1x1000xf16>
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

    config.Resources 3 of @NCE at 6.000000e+02 MHz

    // CHECK: func.func @foo1([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
    func.func @foo1(%arg0: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16> {
        %cst = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
        %cst_0 = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x3x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]
        %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        %1 = VPU.NCE.Permute(%0) {
            dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
        } -> tensor<1x16x62x64xf16, {order = #NHWC}>
        %2 = VPU.Slice %1 [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
        %3 = VPU.NCE.Convolution(%2, %cst_0, %cst) {
                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
                rawFilterShape = [48, 16, 3, 3], strides = [1, 1]}
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
        // CHECK-SAME:       -> !VPU.DistributedTensor<1x3x62x64xf16, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3, 21, 64], [1, 3, 21, 64], [1, 3, 20, 64]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 42, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3, 21, 64], [1, 3, 21, 64], [1, 3, 20, 64]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 42, 0]]}>

        // CHECK:       [[PERM:%.+]] = VPU.NCE.Permute([[COPY0]]
        // CHECK-SAME:       -> !VPU.DistributedTensor<1x16x62x64xf16, #NHWC, @CMX_NN,
        // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 21, 64], [1, 16, 21, 64], [1, 16, 20, 64]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 42, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 21, 64], [1, 16, 21, 64], [1, 16, 20, 64]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 42, 0]]}>

        // CHECK:       [[COPY1:%.+]] = VPU.Copy([[PERM]]
        // CHECK-SAME:       -> tensor<1x16x62x64xf16, {order = #NHWC}>

        // CHECK:       [[SLICE:%.+]] = VPU.Slice [[COPY1]] [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
        // CHECK:       [[IN:%.+]] = VPU.Copy([[SLICE]]
        // CHECK-SAME:       -> !VPU.DistributedTensor<1x16x62x62xf16, #NHWC, @CMX_NN,
        // CHECK-SAME:        {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 21, 62], [1, 16, 21, 62], [1, 16, 20, 62]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 42, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 22, 62], [1, 16, 22, 62], [1, 16, 22, 62]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0], [0, 0, 40, 0]]}>

        // CHECK:       [[COPY2:%.+]] = VPU.Copy([[SPARSE]]
        // CHECK-SAME:       -> !VPU.SparseTensor<data=!VPU.DistributedTensor<48x16x3x3xf16, #NHWC, @CMX_NN,
        // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[48, 16, 3, 3], [48, 16, 3, 3], [48, 16, 3, 3]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[48, 16, 3, 3], [48, 16, 3, 3], [48, 16, 3, 3]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
        // CHECK-SAME:                         sparsity_map=!VPU.DistributedTensor<48x1x1x256xi1, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[48, 1, 1, 256], [48, 1, 1, 256], [48, 1, 1, 256]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[48, 1, 1, 256], [48, 1, 1, 256], [48, 1, 1, 256]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>,
        // CHECK-SAME:                         is_weights, #VPU.SparsityCompression<axis = 0 : i64, numElems = dense<27> : tensor<48xi64>, alignment = 16 : i64>>

        // CHECK:       [[COPY3:%.+]] = VPU.Copy([[CST0]]
        // CHECK-SAME:       -> !VPU.DistributedTensor<48x1x1x4xsi32, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

        // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[IN]], [[COPY2]], [[COPY3]])
        // CHECK-SAME:  -> !VPU.DistributedTensor<1x48x60x60xf16, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 48, 20, 60], [1, 48, 20, 60], [1, 48, 20, 60]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0], [0, 0, 40, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 48, 20, 60], [1, 48, 20, 60], [1, 48, 20, 60]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0], [0, 0, 40, 0]]}>

        // CHECK:       [[OUT:%.+]] = VPU.Copy([[CONV]]
        // CHECK-SAME:       -> tensor<1x48x60x60xf16>

        // CHECK:       return [[OUT]] : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @foo2([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    func.func @foo2(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %0 = VPU.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        return %0 : tensor<1x48x60x60xf16>

        // CHECK:       [[COPY:%.+]] = VPU.Copy([[ARG0]]
        // CHECK-SAME:       -> !VPU.DistributedTensor<1x48x60x60xf16, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 60, 60], [1, 16, 60, 60], [1, 16, 60, 60]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 60, 60], [1, 16, 60, 60], [1, 16, 60, 60]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]

        // CHECK:       [[SOFTMAX:%.+]] = VPU.SoftMax([[COPY]]
        // CHECK-SAME:       -> !VPU.DistributedTensor<1x48x60x60xf16, #NCHW, @CMX_NN,
        // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 60, 60], [1, 16, 60, 60], [1, 16, 60, 60]]
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 60, 60], [1, 16, 60, 60], [1, 16, 60, 60]]
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]

        // CHECK:       [[OUT:%.+]] = VPU.Copy([[SOFTMAX]]
        // CHECK-SAME:      -> tensor<1x48x60x60xf16>

        // CHECK: return [[OUT]] : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func private @main_outline1([[ARG0:%.+]]: tensor<1x3x62x62xui8>) -> tensor<1x3x62x62xf16>
    // CHECK:       [[COPY:%.+]] = VPU.Copy([[ARG0]]
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x3x62x62xui8, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3, 21, 62], [1, 3, 21, 62], [1, 3, 20, 62]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 42, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3, 21, 62], [1, 3, 21, 62], [1, 3, 20, 62]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 42, 0]]}>

    // CHECK:       [[CONVERT:%.+]] = VPU.Convert([[COPY]]
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x3x62x62xf16, #NCHW, @CMX_NN,
    // CHECK-SAME:          {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 3, 21, 62], [1, 3, 21, 62], [1, 3, 20, 62]]
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 42, 0]]
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 3, 21, 62], [1, 3, 21, 62], [1, 3, 20, 62]]
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 21, 0], [0, 0, 42, 0]]}>

    // CHECK:       [[COPY_BACK:%.+]] = VPU.Copy([[CONVERT]]
    // CHECK-SAME:      -> tensor<1x3x62x62xf16>

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

    config.Resources 3 of @NCE at 6.000000e+02 MHz

    // CHECK: func.func private @main_fn1([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
    func.func private @main_fn1(%arg0: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16> {
        %shape_cast1 = VPU.ShapeCast {shape = [1, 48, 225, 16]} inputs(%arg0 : tensor<1x48x60x60xf16>) -> tensor<1x48x225x16xf16>
        %permute = VPU.NCE.Permute(%shape_cast1) {
            dstElemType = f16, dstOrder = #NHWC, expandedChannels = 48 : i64,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>
        } -> tensor<1x48x225x16xf16, {order = #NHWC}>
        %shape_cast2 = VPU.ShapeCast {shape = [1, 48, 60, 60]} inputs(%permute : tensor<1x48x225x16xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>

        %cst_weights_table = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
        %cst_weights = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x48x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
        %conv = VPU.NCE.Convolution(%shape_cast2, %cst_weights, %cst_weights_table) {
            pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>,
            ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
            rawFilterShape = [48, 48, 3, 3], strides = [1, 1]
        } : tensor<1x48x60x60xf16, {order = #NHWC}>, tensor<48x48x3x3xf16, {order = #NHWC}>, tensor<48x1x1x4xsi32> -> tensor<1x48x60x60xf16>

        return %conv : tensor<1x48x60x60xf16>

        // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<48x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x48x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
        // CHECK-DAG:   [[CST_WEIGHTS_TABLE:%.+]] = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>

        // CHECK:       [[SHAPE_CAST1:%.+]] = VPU.ShapeCast {shape = [1, 48, 225, 16]} inputs([[ARG0]] : tensor<1x48x60x60xf16>) -> tensor<1x48x225x16xf16>
        // CHECK:       [[INPUT_COPY1:%.+]] = VPU.Copy([[SHAPE_CAST1]]
        // CHECK-SAME:      -> !VPU.DistributedTensor<1x48x225x16xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 48, 75, 16], [1, 48, 75, 16], [1, 48, 75, 16]],
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 75, 0], [0, 0, 150, 0]],
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 48, 75, 16], [1, 48, 75, 16], [1, 48, 75, 16]],
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 75, 0], [0, 0, 150, 0]]}>

        // CHECK:       [[PERM:%.+]] = VPU.NCE.Permute([[INPUT_COPY1]]
        // CHECK-SAME:      -> !VPU.DistributedTensor<1x48x225x16xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 48, 75, 16], [1, 48, 75, 16], [1, 48, 75, 16]],
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 75, 0], [0, 0, 150, 0]],
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 48, 75, 16], [1, 48, 75, 16], [1, 48, 75, 16]],
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 75, 0], [0, 0, 150, 0]]}>

        // CHECK:       [[INPUT_COPY2:%.+]] = VPU.Copy([[PERM]]
        // CHECK-SAME:       -> tensor<1x48x225x16xf16, {order = #NHWC}>

        // CHECK:       [[SHAPE_CAST2:%.+]] = VPU.ShapeCast {shape = [1, 48, 60, 60]} inputs([[INPUT_COPY2]] : tensor<1x48x225x16xf16, {order = #NHWC}>) -> tensor<1x48x60x60xf16, {order = #NHWC}>
        // CHECK:       [[INPUT_COPY3:%.+]] = VPU.Copy([[SHAPE_CAST2]]
        // CHECK-SAME:      -> !VPU.DistributedTensor<1x48x60x60xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 48, 20, 60], [1, 48, 20, 60], [1, 48, 20, 60]],
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0], [0, 0, 40, 0]],
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 48, 21, 60], [1, 48, 22, 60], [1, 48, 21, 60]],
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 19, 0], [0, 0, 39, 0]]}>

        // CHECK:       [[COPY_WEIGHTS:%.+]] = VPU.Copy([[CST_WEIGHTS]]
        // CHECK-SAME:      -> !VPU.DistributedTensor<48x48x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
        // CHECK-SAME{LITERAL}:  compute_shapes = [[48, 48, 3, 3], [48, 48, 3, 3], [48, 48, 3, 3]],
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        // CHECK-SAME{LITERAL}:  memory_shapes = [[48, 48, 3, 3], [48, 48, 3, 3], [48, 48, 3, 3]],
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

        // CHECK:       [[COPY_WEIGHTS_TABLE:%.+]] = VPU.Copy([[CST_WEIGHTS_TABLE]]
        // CHECK-SAME:      -> !VPU.DistributedTensor<48x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,
        // CHECK-SAME{LITERAL}:  compute_shapes = [[48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4]],
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        // CHECK-SAME{LITERAL}:  memory_shapes = [[48, 1, 1, 4], [48, 1, 1, 4], [48, 1, 1, 4]],
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}>

        // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT_COPY3]], [[COPY_WEIGHTS]], [[COPY_WEIGHTS_TABLE]])
        // CHECK-SAME:  -> !VPU.DistributedTensor<1x48x60x60xf16, #NCHW, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
        // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 48, 20, 60], [1, 48, 20, 60], [1, 48, 20, 60]],
        // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 0, 20, 0], [0, 0, 40, 0]],
        // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 48, 20, 60], [1, 48, 20, 60], [1, 48, 20, 60]],
        // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 0, 20, 0], [0, 0, 40, 0]]}>

        // CHECK:       [[OUTPUT_COPY:%.+]] = VPU.Copy([[CONV]]
        // CHECK-SAME:       -> tensor<1x48x60x60xf16>

        // CHECK:       return [[OUTPUT_COPY]]
    }

    // CHECK: func.func private @main_outline1([[INPUT:%.+]]: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf16> {
    // CHECK:       [[COPY:%.+]] = VPU.Copy([[INPUT]]
    // CHECK-SAME:       -> !VPU.DistributedTensor<1x48x60x60xf32, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 60, 60], [1, 16, 60, 60], [1, 16, 60, 60]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 60, 60], [1, 16, 60, 60], [1, 16, 60, 60]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]}>

    // CHECK:       [[CONVERT:%.+]] = VPU.Convert([[COPY]]
    // CHECK-SAME:      -> !VPU.DistributedTensor<1x48x60x60xf16, #NCHW, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 3, 1, 1], num_clusters = 3 : i64, uniform_distributed_segments,
    // CHECK-SAME{LITERAL}:  compute_shapes = [[1, 16, 60, 60], [1, 16, 60, 60], [1, 16, 60, 60]],
    // CHECK-SAME{LITERAL}:  compute_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]],
    // CHECK-SAME{LITERAL}:  memory_shapes = [[1, 16, 60, 60], [1, 16, 60, 60], [1, 16, 60, 60]],
    // CHECK-SAME{LITERAL}:  memory_offsets = [[0, 0, 0, 0], [0, 16, 0, 0], [0, 32, 0, 0]]}>

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
        DataInfo "input" : tensor<1x30x256x256xf32, {order = #NHWC}>
    } outputsInfo : {
        DataInfo "output" : tensor<1x32x256x256xf16, {order = #NHWC}>
    }

    func.func @main(%arg0: tensor<1x30x256x256xf32, {order = #NHWC}>) -> tensor<1x32x256x256xf16, {order = #NHWC}> {
        %cst = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
        %cst_0 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

        %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x30x256x256xf32, {order = #NHWC}> -> tensor<1x30x256x256xf16, {order = #NHWC}>
        %1 = VPU.Expand(%0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 2, 0, 0]} : tensor<1x30x256x256xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>

        %2 = VPU.NCE.Convolution(%1, %cst_0, %cst) {
                ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e-01], adder = 0.000000e+00 : f64>,
                pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, rawFilterShape = [32, 32, 3, 3], strides = [1, 1]}
                    : tensor<1x32x256x256xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x256x256xf16, {order = #NHWC}>
        %3 = VPU.SoftMax(%2) {axisInd = 3 : i64} : tensor<1x32x256x256xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>

        return %3  : tensor<1x32x256x256xf16, {order = #NHWC}>
    }
}

// CHECK:     func.func private @main_vf1([[ARG0:%.+]]: tensor<1x30x256x256xf32, {order = #NHWC}>) -> tensor<1x30x256x256xf16, {order = #NHWC}> {
// CHECK:       [[SLICE_0:%.+]] = VPU.Slice [[ARG0]] [0, 0, 0, 0] [1, 30, 86, 256] : tensor<1x30x256x256xf32, {order = #NHWC}> to tensor<1x30x86x256xf32, {order = #NHWC}>
// CHECK:       [[COPY0_0:%.+]] = VPU.Copy([[SLICE_0]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x30x86x256xf32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[CONVERT0:%.+]] = VPU.Convert([[COPY0_0]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x30x86x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[COPY0_1:%.+]] = VPU.Copy([[CONVERT0]]
// CHECK-SAME:       -> tensor<1x30x86x256xf16, {order = #NHWC}>

// CHECK:       [[SLICE_1:%.+]] = VPU.Slice [[ARG0]] [0, 0, 86, 0] [1, 30, 85, 256] : tensor<1x30x256x256xf32, {order = #NHWC}> to tensor<1x30x85x256xf32, {order = #NHWC}>
// CHECK:       [[COPY1_0:%.+]] = VPU.Copy([[SLICE_1]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x30x85x256xf32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[CONVERT1:%.+]] = VPU.Convert([[COPY1_0]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x30x85x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[COPY1_1:%.+]] = VPU.Copy([[CONVERT1]]
// CHECK-SAME:       -> tensor<1x30x85x256xf16, {order = #NHWC}>

// CHECK:       [[SLICE_2:%.+]] = VPU.Slice [[ARG0]] [0, 0, 171, 0] [1, 30, 85, 256] : tensor<1x30x256x256xf32, {order = #NHWC}> to tensor<1x30x85x256xf32, {order = #NHWC}>
// CHECK:       [[COPY2_0:%.+]] = VPU.Copy([[SLICE_2]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x30x85x256xf32, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[CONVERT2:%.+]] = VPU.Convert([[COPY2_0]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x30x85x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[COPY2_1:%.+]] = VPU.Copy([[CONVERT2]]
// CHECK-SAME:       -> tensor<1x30x85x256xf16, {order = #NHWC}>

// CHECK:       [[CONCAT:%.+]] = VPU.Concat([[COPY0_1]], [[COPY1_1]], [[COPY2_1]])
// CHECK-SAME:  : tensor<1x30x86x256xf16, {order = #NHWC}>, tensor<1x30x85x256xf16, {order = #NHWC}>, tensor<1x30x85x256xf16, {order = #NHWC}> -> tensor<1x30x256x256xf16, {order = #NHWC}>
// CHECK:       return [[CONCAT]] : tensor<1x30x256x256xf16, {order = #NHWC}>

// CHECK:     func.func private @main_vf2([[ARG0:%.+]]: tensor<1x30x256x256xf16, {order = #NHWC}>) -> tensor<1x32x256x256xf16, {order = #NHWC}> attributes {pure_vertical_fusion_region} {
// CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<32x1x1x4xsi32> = dense_resource<__elided__> : tensor<32x1x1x4xsi32>
// CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x32x3x3xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
// CHECK:       [[EXAPND:%.+]] = VPU.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 2, 0, 0]} : tensor<1x30x256x256xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>

// CHECK:       [[SLICE_0:%.+]] = VPU.Slice [[EXAPND]] [0, 0, 0, 0] [1, 32, 53, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x32x53x256xf16, {order = #NHWC}>
// CHECK:       [[COPY0_0:%.+]] = VPU.Copy([[SLICE_0]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x32x53x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[COPY0_1:%.+]] = VPU.Copy([[CST_0]]
// CHECK-SAME:       -> !VPU.DistributedTensor<32x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[COPY0_2:%.+]] = VPU.Copy([[CST]]
// CHECK-SAME:       -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[CONV0:%.+]] = VPU.NCE.Convolution([[COPY0_0]], [[COPY0_1]], [[COPY0_2]])
// CHECK-SAME:       -> !VPU.DistributedTensor<1x32x52x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[CAST0:%.+]] = VPU.DistributedCast([[CONV0]] : !VPU.DistributedTensor<1x32x52x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK:       [[SOFTMAX0:%.+]] = VPU.SoftMax([[CAST0]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x32x52x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[COPY0_3:%.+]] = VPU.Copy([[SOFTMAX0]]
// CHECK-SAME:       -> tensor<1x32x52x256xf16, {order = #NHWC}>

// CHECK:       [[SLICE_1:%.+]] = VPU.Slice [[EXAPND]] [0, 0, 51, 0] [1, 32, 53, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x32x53x256xf16, {order = #NHWC}>
// CHECK:       [[COPY1_0:%.+]] = VPU.Copy([[SLICE_1]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x32x53x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[COPY1_1:%.+]] = VPU.Copy([[CST_0]]
// CHECK-SAME:       -> !VPU.DistributedTensor<32x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[COPY1_2:%.+]] = VPU.Copy([[CST]]
// CHECK-SAME:       -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[COPY1_0]], [[COPY1_1]], [[COPY1_2]])
// CHECK-SAME:       -> !VPU.DistributedTensor<1x32x51x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[CAST1:%.+]] = VPU.DistributedCast([[CONV1]] : !VPU.DistributedTensor<1x32x51x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK:       [[SOFTMAX1:%.+]] = VPU.SoftMax([[CAST1]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x32x51x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[COPY1_3:%.+]] = VPU.Copy([[SOFTMAX1]]
// CHECK-SAME:       -> tensor<1x32x51x256xf16, {order = #NHWC}>

// CHECK:       [[SLICE_2:%.+]] = VPU.Slice [[EXAPND]] [0, 0, 102, 0] [1, 32, 53, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x32x53x256xf16, {order = #NHWC}>
// CHECK:       [[COPY2_0:%.+]] = VPU.Copy([[SLICE_2]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x32x53x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[COPY2_1:%.+]] = VPU.Copy([[CST_0]]
// CHECK-SAME:       -> !VPU.DistributedTensor<32x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[COPY2_2:%.+]] = VPU.Copy([[CST]]
// CHECK-SAME:       -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[COPY2_0]], [[COPY2_1]], [[COPY2_2]])
// CHECK-SAME:       -> !VPU.DistributedTensor<1x32x51x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[CAST2:%.+]] = VPU.DistributedCast([[CONV2]] : !VPU.DistributedTensor<1x32x51x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK:       [[SOFTMAX2:%.+]] = VPU.SoftMax([[CAST2]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x32x51x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[COPY2_3:%.+]] = VPU.Copy([[SOFTMAX2]]
// CHECK-SAME:       -> tensor<1x32x51x256xf16, {order = #NHWC}>

// CHECK:       [[SLICE_3:%.+]] = VPU.Slice [[EXAPND]] [0, 0, 153, 0] [1, 32, 53, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x32x53x256xf16, {order = #NHWC}>
// CHECK:       [[COPY3_0:%.+]] = VPU.Copy([[SLICE_3]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x32x53x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[COPY3_1:%.+]] = VPU.Copy([[CST_0]]
// CHECK-SAME:       -> !VPU.DistributedTensor<32x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[COPY3_2:%.+]] = VPU.Copy([[CST]]
// CHECK-SAME:       -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[CONV3:%.+]] = VPU.NCE.Convolution([[COPY3_0]], [[COPY3_1]], [[COPY3_2]])
// CHECK-SAME:       -> !VPU.DistributedTensor<1x32x51x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[CAST3:%.+]] = VPU.DistributedCast([[CONV3]] : !VPU.DistributedTensor<1x32x51x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK:       [[SOFTMAX3:%.+]] = VPU.SoftMax([[CAST3]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x32x51x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[COPY3_3:%.+]] = VPU.Copy([[SOFTMAX3]]
// CHECK-SAME:       -> tensor<1x32x51x256xf16, {order = #NHWC}>

// CHECK:       [[SLICE_4:%.+]] = VPU.Slice [[EXAPND]] [0, 0, 204, 0] [1, 32, 52, 256] : tensor<1x32x256x256xf16, {order = #NHWC}> to tensor<1x32x52x256xf16, {order = #NHWC}>
// CHECK:       [[COPY4_0:%.+]] = VPU.Copy([[SLICE_4]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x32x52x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[COPY4_1:%.+]] = VPU.Copy([[CST_0]]
// CHECK-SAME:       -> !VPU.DistributedTensor<32x32x3x3xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[COPY4_2:%.+]] = VPU.Copy([[CST]]
// CHECK-SAME:       -> !VPU.DistributedTensor<32x1x1x4xsi32, #NCHW, @CMX_NN, {mode = "DUPLICATED", num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[CONV4:%.+]] = VPU.NCE.Convolution([[COPY4_0]], [[COPY4_1]], [[COPY4_2]])
// CHECK-SAME:       -> !VPU.DistributedTensor<1x32x51x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[CAST4:%.+]] = VPU.DistributedCast([[CONV4]] : !VPU.DistributedTensor<1x32x51x256xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,
// CHECK:       [[SOFTMAX4:%.+]] = VPU.SoftMax([[CAST4]]
// CHECK-SAME:       -> !VPU.DistributedTensor<1x32x51x256xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 3, 1], num_clusters = 3 : i64, uniform_distributed_segments,

// CHECK:       [[COPY4_3:%.+]] = VPU.Copy([[SOFTMAX4]]
// CHECK-SAME:       -> tensor<1x32x51x256xf16, {order = #NHWC}>

// CHECK:       [[CONCAT:%.+]] = VPU.Concat([[COPY0_3]], [[COPY1_3]], [[COPY2_3]], [[COPY3_3]], [[COPY4_3]])
// CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 0, 52, 0], [0, 0, 103, 0], [0, 0, 154, 0], [0, 0, 205, 0]]}
// CHECK-SAME:           : tensor<1x32x52x256xf16, {order = #NHWC}>, tensor<1x32x51x256xf16, {order = #NHWC}>, tensor<1x32x51x256xf16, {order = #NHWC}>, tensor<1x32x51x256xf16, {order = #NHWC}>, tensor<1x32x51x256xf16, {order = #NHWC}> -> tensor<1x32x256x256xf16, {order = #NHWC}>
// CHECK:       return [[CONCAT]] : tensor<1x32x256x256xf16, {order = #NHWC}>

// CHECK:     func.func @main([[ARG0:%.+]]: tensor<1x30x256x256xf32, {order = #NHWC}>) -> tensor<1x32x256x256xf16, {order = #NHWC}> {
// CHECK:       [[CALL0:%.+]] = call @main_vf1([[ARG0]]) : (tensor<1x30x256x256xf32, {order = #NHWC}>) -> tensor<1x30x256x256xf16, {order = #NHWC}>
// CHECK:       [[CALL1:%.+]] = call @main_vf2([[CALL0]]) : (tensor<1x30x256x256xf16, {order = #NHWC}>) -> tensor<1x32x256x256xf16, {order = #NHWC}>
// CHECK:       return [[CALL1]] : tensor<1x32x256x256xf16, {order = #NHWC}>

// -----

// CHECK-LABEL: @AdjustMemorySpaceAndOptimizeSharedInputCopyForConcat1T
// Check whether OptimizeSharedInputCopyForConcat is applied after reordering with AdjustMemorySpace on 1T (E#156584)
// The pattern that OptimizeSharedInputCopyForConcat expects: Copy(CMX2DDR) -> Concat(DDR) -> {Concat(DDR) -> Slice(DDR) -> Copy(DDR2CMX)} x N, where N >= 2
// On 1 Tile, CMX won't be involved until AdjustMemorySpace, so OptimizeSharedInputCopyForConcat will fail to match due to the missing CopyOps at the beginning and end
// By bringing AdjustMemorySpace in front of OptimizeSharedInputCopyForConcat, the pattern will be matched and rewritten to: Copy(CMX2DDR) -> Concat(DDR) -> {Slice(DDR) -> Copy(DDR2CMX) -> Concat(CMX)}
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @AdjustMemorySpaceAndOptimizeSharedInputCopyForConcat1T {
  config.Resources 1 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 1 of @M2I
  config.ExecutorResource 1 of @DMA_NN
  config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
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
    %1 = VPU.NCE.Permute(%0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 3 : i64, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x3x128x128xf16, {order = #NHWC}>
    %2 = VPU.ShapeCast {shape = [1, 48, 128, 8]} inputs(%1 : tensor<1x3x128x128xf16, {order = #NHWC}>) -> tensor<1x48x128x8xf16, {order = #NHWC}>
    %cst_6 = const.Declare tensor<1024x1x1x4xsi32> = dense<1> : tensor<1024x1x1x4xsi32>
    %3 = VPU.NCE.Convolution(%2, %cst, %cst_6) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [1024, 48, 3, 3], strides = [1, 1]} : tensor<1x48x128x8xf16, {order = #NHWC}>, tensor<1024x48x3x3xf16, {order = #NHWC}>, tensor<1024x1x1x4xsi32> -> tensor<1x1024x128x8xf16, {order = #NHWC}>
    %4 = VPU.ShapeCast {shape = [1, 64, 128, 128]} inputs(%3 : tensor<1x1024x128x8xf16, {order = #NHWC}>) -> tensor<1x64x128x128xf16, {order = #NHWC}>
    %cst_7 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %5 = VPU.NCE.Convolution(%4, %cst_3, %cst_7) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [0.199951171875], adder = 0.000000e+00 : f64>, rawFilterShape = [32, 64, 3, 3], strides = [1, 1]} : tensor<1x64x128x128xf16, {order = #NHWC}>, tensor<32x64x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x128x128xf16, {order = #NHWC}>
    %6 = VPU.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x128x128xf16, {order = #NHWC}>, tensor<1x32x128x128xf16, {order = #NHWC}> -> tensor<1x96x128x128xf16, {order = #NHWC}>
    %cst_8 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %7 = VPU.NCE.Convolution(%6, %cst_2, %cst_8) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [0.199951171875], adder = 0.000000e+00 : f64>, rawFilterShape = [32, 96, 3, 3], strides = [1, 1]} : tensor<1x96x128x128xf16, {order = #NHWC}>, tensor<32x96x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x128x128xf16, {order = #NHWC}>
    %8 = VPU.Concat(%4, %5, %7) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]} : tensor<1x64x128x128xf16, {order = #NHWC}>, tensor<1x32x128x128xf16, {order = #NHWC}>, tensor<1x32x128x128xf16, {order = #NHWC}> -> tensor<1x128x128x128xf16, {order = #NHWC}>
    %cst_9 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %9 = VPU.NCE.Convolution(%8, %cst_1, %cst_9) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, rawFilterShape = [32, 128, 3, 3], strides = [1, 1]} : tensor<1x128x128x128xf16, {order = #NHWC}>, tensor<32x128x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x128x128xf16>
    %10 = VPU.Convert(%9) {dstElemType = f32} : tensor<1x32x128x128xf16> -> tensor<1x32x128x128xf32>
    return %10 : tensor<1x32x128x128xf32>

    // CHECK:   func.func @main([[ARG0:%.+]]: tensor<1x3x128x128xf32>) -> tensor<1x32x128x128xf32> {
        // CHECK:       [[SHAPECAST1:%.+]] = VPU.ShapeCast {shape = [1, 64, 128, 128]} inputs({{%.+}} : tensor<1x1024x128x8xf16, {order = #NHWC}>) -> tensor<1x64x128x128xf16, {order = #NHWC}>
        // CHECK:       [[CONCAT1:%.+]] = VPU.Concat({{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}) {static_offsets =
        // CHECK-SAME{LITERAL}:   [[0, 0, 0, 0], [0, 0, 0, 22], [0, 0, 0, 44], [0, 0, 0, 65], [0, 0, 0, 86], [0, 0, 0, 107]]

        // CHECK:       [[SLICE10:%.+]] = VPU.Slice [[CONCAT1]] [0, 0, 0, 0] [1, 32, 128, 16] : tensor<1x32x128x128xf16, {order = #NHWC}> to tensor<1x32x128x16xf16, {order = #NHWC}>
        // CHECK:       [[COPY10:%.+]] = VPU.Copy([[SLICE10]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x128x16xf16, {order = #NHWC}> -> tensor<1x32x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE11:%.+]] = VPU.Slice [[SHAPECAST1]] [0, 0, 0, 0] [1, 64, 128, 16] : tensor<1x64x128x128xf16, {order = #NHWC}> to tensor<1x64x128x16xf16, {order = #NHWC}>
        // CHECK:       [[COPY11:%.+]] = VPU.Copy([[SLICE11]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x64x128x16xf16, {order = #NHWC}> -> tensor<1x64x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE12:%.+]] = VPU.Slice [[CONCAT1]] [0, 0, 0, 14] [1, 32, 128, 17] : tensor<1x32x128x128xf16, {order = #NHWC}> to tensor<1x32x128x17xf16, {order = #NHWC}>
        // CHECK:       [[COPY12:%.+]] = VPU.Copy([[SLICE12]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x128x17xf16, {order = #NHWC}> -> tensor<1x32x128x17xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE13:%.+]] = VPU.Slice [[SHAPECAST1]] [0, 0, 0, 14] [1, 64, 128, 17] : tensor<1x64x128x128xf16, {order = #NHWC}> to tensor<1x64x128x17xf16, {order = #NHWC}>
        // CHECK:       [[COPY13:%.+]] = VPU.Copy([[SLICE13]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x64x128x17xf16, {order = #NHWC}> -> tensor<1x64x128x17xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE14:%.+]] = VPU.Slice [[CONCAT1]] [0, 0, 0, 29] [1, 32, 128, 16] : tensor<1x32x128x128xf16, {order = #NHWC}> to tensor<1x32x128x16xf16, {order = #NHWC}>
        // CHECK:       [[COPY14:%.+]] = VPU.Copy([[SLICE14]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x128x16xf16, {order = #NHWC}> -> tensor<1x32x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE15:%.+]] = VPU.Slice [[SHAPECAST1]] [0, 0, 0, 29] [1, 64, 128, 16] : tensor<1x64x128x128xf16, {order = #NHWC}> to tensor<1x64x128x16xf16, {order = #NHWC}>
        // CHECK:       [[COPY15:%.+]] = VPU.Copy([[SLICE15]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x64x128x16xf16, {order = #NHWC}> -> tensor<1x64x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE16:%.+]] = VPU.Slice [[CONCAT1]] [0, 0, 0, 43] [1, 32, 128, 16] : tensor<1x32x128x128xf16, {order = #NHWC}> to tensor<1x32x128x16xf16, {order = #NHWC}>
        // CHECK:       [[COPY16:%.+]] = VPU.Copy([[SLICE16]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x128x16xf16, {order = #NHWC}> -> tensor<1x32x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE17:%.+]] = VPU.Slice [[SHAPECAST1]] [0, 0, 0, 43] [1, 64, 128, 16] : tensor<1x64x128x128xf16, {order = #NHWC}> to tensor<1x64x128x16xf16, {order = #NHWC}>
        // CHECK:       [[COPY17:%.+]] = VPU.Copy([[SLICE17]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x64x128x16xf16, {order = #NHWC}> -> tensor<1x64x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE18:%.+]] = VPU.Slice [[CONCAT1]] [0, 0, 0, 57] [1, 32, 128, 16] : tensor<1x32x128x128xf16, {order = #NHWC}> to tensor<1x32x128x16xf16, {order = #NHWC}>
        // CHECK:       [[COPY18:%.+]] = VPU.Copy([[SLICE18]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x128x16xf16, {order = #NHWC}> -> tensor<1x32x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE19:%.+]] = VPU.Slice [[SHAPECAST1]] [0, 0, 0, 57] [1, 64, 128, 16] : tensor<1x64x128x128xf16, {order = #NHWC}> to tensor<1x64x128x16xf16, {order = #NHWC}>
        // CHECK:       [[COPY19:%.+]] = VPU.Copy([[SLICE19]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x64x128x16xf16, {order = #NHWC}> -> tensor<1x64x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE20:%.+]] = VPU.Slice [[CONCAT1]] [0, 0, 0, 71] [1, 32, 128, 16] : tensor<1x32x128x128xf16, {order = #NHWC}> to tensor<1x32x128x16xf16, {order = #NHWC}>
        // CHECK:       [[COPY20:%.+]] = VPU.Copy([[SLICE20]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x128x16xf16, {order = #NHWC}> -> tensor<1x32x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE21:%.+]] = VPU.Slice [[SHAPECAST1]] [0, 0, 0, 71] [1, 64, 128, 16] : tensor<1x64x128x128xf16, {order = #NHWC}> to tensor<1x64x128x16xf16, {order = #NHWC}>
        // CHECK:       [[COPY21:%.+]] = VPU.Copy([[SLICE21]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x64x128x16xf16, {order = #NHWC}> -> tensor<1x64x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE22:%.+]] = VPU.Slice [[CONCAT1]] [0, 0, 0, 85] [1, 32, 128, 16] : tensor<1x32x128x128xf16, {order = #NHWC}> to tensor<1x32x128x16xf16, {order = #NHWC}>
        // CHECK:       [[COPY22:%.+]] = VPU.Copy([[SLICE22]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x128x16xf16, {order = #NHWC}> -> tensor<1x32x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE23:%.+]] = VPU.Slice [[SHAPECAST1]] [0, 0, 0, 85] [1, 64, 128, 16] : tensor<1x64x128x128xf16, {order = #NHWC}> to tensor<1x64x128x16xf16, {order = #NHWC}>
        // CHECK:       [[COPY23:%.+]] = VPU.Copy([[SLICE23]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x64x128x16xf16, {order = #NHWC}> -> tensor<1x64x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE24:%.+]] = VPU.Slice [[CONCAT1]] [0, 0, 0, 99] [1, 32, 128, 16] : tensor<1x32x128x128xf16, {order = #NHWC}> to tensor<1x32x128x16xf16, {order = #NHWC}>
        // CHECK:       [[COPY24:%.+]] = VPU.Copy([[SLICE24]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x128x16xf16, {order = #NHWC}> -> tensor<1x32x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE25:%.+]] = VPU.Slice [[SHAPECAST1]] [0, 0, 0, 99] [1, 64, 128, 16] : tensor<1x64x128x128xf16, {order = #NHWC}> to tensor<1x64x128x16xf16, {order = #NHWC}>
        // CHECK:       [[COPY25:%.+]] = VPU.Copy([[SLICE25]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x64x128x16xf16, {order = #NHWC}> -> tensor<1x64x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE26:%.+]] = VPU.Slice [[CONCAT1]] [0, 0, 0, 113] [1, 32, 128, 15] : tensor<1x32x128x128xf16, {order = #NHWC}> to tensor<1x32x128x15xf16, {order = #NHWC}>
        // CHECK:       [[COPY26:%.+]] = VPU.Copy([[SLICE26]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x32x128x15xf16, {order = #NHWC}> -> tensor<1x32x128x15xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[SLICE27:%.+]] = VPU.Slice [[SHAPECAST1]] [0, 0, 0, 113] [1, 64, 128, 15] : tensor<1x64x128x128xf16, {order = #NHWC}> to tensor<1x64x128x15xf16, {order = #NHWC}>
        // CHECK:       [[COPY27:%.+]] = VPU.Copy([[SLICE27]]) {out_mem_space = [@CMX_NN, 0]} : tensor<1x64x128x15xf16, {order = #NHWC}> -> tensor<1x64x128x15xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[CONCAT2:%.+]] = VPU.Concat([[COPY27]], [[COPY26]])
        // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x128x15xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x128x15xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x96x128x15xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[CONCAT3:%.+]] = VPU.Concat([[COPY25]], [[COPY24]])
        // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x96x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[CONCAT4:%.+]] = VPU.Concat([[COPY23]], [[COPY22]])
        // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x96x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[CONCAT5:%.+]] = VPU.Concat([[COPY21]], [[COPY20]])
        // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x96x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[CONCAT6:%.+]] = VPU.Concat([[COPY19]], [[COPY18]])
        // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x96x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[CONCAT7:%.+]] = VPU.Concat([[COPY17]], [[COPY16]])
        // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x96x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[CONCAT8:%.+]] = VPU.Concat([[COPY15]], [[COPY14]])
        // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x96x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[CONCAT9:%.+]] = VPU.Concat([[COPY13]], [[COPY12]])
        // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x128x17xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x128x17xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x96x128x17xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>
        // CHECK:       [[CONCAT10:%.+]] = VPU.Concat([[COPY11]], [[COPY10]])
        // CHECK-SAME{LITERAL}: {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>, tensor<1x32x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}> -> tensor<1x96x128x16xf16, {mem_space = [@CMX_NN, 0], order = #NHWC}>

        // CHECK:       return {{%.+}} : tensor<1x32x128x128xf32>

  }
}

// -----

// CHECK-LABEL: @AdjustMemorySpaceAndOptimizeSharedInputCopyForConcat2T
// Check whether OptimizeSharedInputCopyForConcat is applied after reordering with AdjustMemorySpace on 2T (E#156584)
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @AdjustMemorySpaceAndOptimizeSharedInputCopyForConcat2T {
  config.Resources 2 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.ExecutorResource 1 of @M2I
  config.ExecutorResource 2 of @DMA_NN
  config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" tensorNames = ["input"] : tensor<1x3x128x128xf32>
  } outputsInfo : {
    DataInfo "output" friendlyName = "output" : tensor<1x32x128x128xf32>
  }
// CHECK-LABEL:   func.func @main
// CHECK-SAME       [[ARG0:%.+]]: tensor<1x3x128x128xf32>
  func.func @main(%arg0: tensor<1x3x128x128xf32>) -> tensor<1x32x128x128xf32> {
    %cst = const.Declare tensor<1024x48x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x144x1x3xf16, {order = #NHWC}>, [#const.Reshape<[1024, 48, 3, 3]>]
    %cst_0 = const.Declare tensor<1x1024x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf32>, [#const.CastElemType<f16>, #const.Broadcast<1 : i64, 1024 : i64>, #const.Reshape<[1, 1024, 1, 1]>]
    %cst_1 = const.Declare tensor<32x128x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x128x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<32x96x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x96x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<32x64x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x64x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf32>, [#const.CastElemType<f16>]
    %cst_5 = const.Declare tensor<1x32x1x1xf16> = dense<1.000000e+00> : tensor<1x32x1x1xf32>, [#const.CastElemType<f16>]
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x3x128x128xf32> -> tensor<1x3x128x128xf16>
    %1 = VPU.NCE.Permute(%0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 3 : i64, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>} -> tensor<1x3x128x128xf16, {order = #NHWC}>
    %2 = VPU.ShapeCast {shape = [1, 48, 128, 8]} inputs(%1 : tensor<1x3x128x128xf16, {order = #NHWC}>) -> tensor<1x48x128x8xf16, {order = #NHWC}>
    %cst_6 = const.Declare tensor<1024x1x1x4xsi32> = dense<1> : tensor<1024x1x1x4xsi32>
    %3 = VPU.NCE.Convolution(%2, %cst, %cst_6) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [1024, 48, 3, 3], strides = [1, 1]} : tensor<1x48x128x8xf16, {order = #NHWC}>, tensor<1024x48x3x3xf16, {order = #NHWC}>, tensor<1024x1x1x4xsi32> -> tensor<1x1024x128x8xf16, {order = #NHWC}>
    %4 = VPU.ShapeCast {shape = [1, 64, 128, 128]} inputs(%3 : tensor<1x1024x128x8xf16, {order = #NHWC}>) -> tensor<1x64x128x128xf16, {order = #NHWC}>
    %cst_7 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %5 = VPU.NCE.Convolution(%4, %cst_3, %cst_7) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [0.199951171875], adder = 0.000000e+00 : f64>, rawFilterShape = [32, 64, 3, 3], strides = [1, 1]} : tensor<1x64x128x128xf16, {order = #NHWC}>, tensor<32x64x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x128x128xf16, {order = #NHWC}>
    %6 = VPU.Concat(%4, %5) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x128x128xf16, {order = #NHWC}>, tensor<1x32x128x128xf16, {order = #NHWC}> -> tensor<1x96x128x128xf16, {order = #NHWC}>
    %cst_8 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %7 = VPU.NCE.Convolution(%6, %cst_2, %cst_8) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [0.199951171875], adder = 0.000000e+00 : f64>, rawFilterShape = [32, 96, 3, 3], strides = [1, 1]} : tensor<1x96x128x128xf16, {order = #NHWC}>, tensor<32x96x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x128x128xf16, {order = #NHWC}>
    %8 = VPU.Concat(%4, %5, %7) {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0], [0, 96, 0, 0]]} : tensor<1x64x128x128xf16, {order = #NHWC}>, tensor<1x32x128x128xf16, {order = #NHWC}>, tensor<1x32x128x128xf16, {order = #NHWC}> -> tensor<1x128x128x128xf16, {order = #NHWC}>
    %cst_9 = const.Declare tensor<32x1x1x4xsi32> = dense<1> : tensor<32x1x1x4xsi32>
    %9 = VPU.NCE.Convolution(%8, %cst_1, %cst_9) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, rawFilterShape = [32, 128, 3, 3], strides = [1, 1]} : tensor<1x128x128x128xf16, {order = #NHWC}>, tensor<32x128x3x3xf16, {order = #NHWC}>, tensor<32x1x1x4xsi32> -> tensor<1x32x128x128xf16>
    %10 = VPU.Convert(%9) {dstElemType = f32} : tensor<1x32x128x128xf16> -> tensor<1x32x128x128xf32>
    return %10 : tensor<1x32x128x128xf32>

        // CHECK:       [[SHAPECAST1:%.+]] = VPU.ShapeCast {shape = [1, 64, 128, 128]} inputs({{%.+}} : tensor<1x1024x128x8xf16, {order = #NHWC}>) -> tensor<1x64x128x128xf16, {order = #NHWC}>
        // CHECK:       [[CONCAT1:%.+]] = VPU.Concat({{%.+}}, {{%.+}}, {{%.+}}) {static_offsets =
        // CHECK-SAME{LITERAL}:  [[0, 0, 0, 0], [0, 0, 0, 43], [0, 0, 0, 86]]} : tensor<1x32x128x43xf16, {order = #NHWC}>, tensor<1x32x128x43xf16, {order = #NHWC}>, tensor<1x32x128x42xf16, {order = #NHWC}> -> tensor<1x32x128x128xf16, {order = #NHWC}>

        // CHECK:       [[SLICE0:%.+]] = VPU.Slice [[CONCAT1]] [0, 0, 0, 0] [1, 32, 128, 27] : tensor<1x32x128x128xf16, {order = #NHWC}> to tensor<1x32x128x27xf16, {order = #NHWC}>
        // CHECK:       [[COPY0:%.+]] = VPU.Copy([[SLICE0]])
        // CHECK-SAME{LITERAL}: {out_mem_space = @CMX_NN} : tensor<1x32x128x27xf16, {order = #NHWC}> -> !VPU.DistributedTensor<1x32x128x27xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 65, 27], [1, 32, 65, 27]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]], memory_shapes = [[1, 32, 65, 27], [1, 32, 65, 27]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}>
        // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[SHAPECAST1]] [0, 0, 0, 0] [1, 64, 128, 27] : tensor<1x64x128x128xf16, {order = #NHWC}> to tensor<1x64x128x27xf16, {order = #NHWC}>
        // CHECK:       [[COPY1:%.+]] = VPU.Copy([[SLICE1]])
        // CHECK-SAME{LITERAL}: {out_mem_space = @CMX_NN} : tensor<1x64x128x27xf16, {order = #NHWC}> -> !VPU.DistributedTensor<1x64x128x27xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 64, 65, 27], [1, 64, 65, 27]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]], memory_shapes = [[1, 64, 65, 27], [1, 64, 65, 27]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}>
        // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[CONCAT1]] [0, 0, 0, 25] [1, 32, 128, 28] : tensor<1x32x128x128xf16, {order = #NHWC}> to tensor<1x32x128x28xf16, {order = #NHWC}>
        // CHECK:       [[COPY2:%.+]] = VPU.Copy([[SLICE2]])
        // CHECK-SAME{LITERAL}: {out_mem_space = @CMX_NN} : tensor<1x32x128x28xf16, {order = #NHWC}> -> !VPU.DistributedTensor<1x32x128x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 65, 28], [1, 32, 65, 28]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]], memory_shapes = [[1, 32, 65, 28], [1, 32, 65, 28]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}>
        // CHECK:       [[SLICE3:%.+]] = VPU.Slice [[SHAPECAST1]] [0, 0, 0, 25] [1, 64, 128, 28] : tensor<1x64x128x128xf16, {order = #NHWC}> to tensor<1x64x128x28xf16, {order = #NHWC}>
        // CHECK:       [[COPY3:%.+]] = VPU.Copy([[SLICE3]])
        // CHECK-SAME{LITERAL}: {out_mem_space = @CMX_NN} : tensor<1x64x128x28xf16, {order = #NHWC}> -> !VPU.DistributedTensor<1x64x128x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 64, 65, 28], [1, 64, 65, 28]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]], memory_shapes = [[1, 64, 65, 28], [1, 64, 65, 28]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}>
        // CHECK:       [[SLICE4:%.+]] = VPU.Slice [[CONCAT1]] [0, 0, 0, 51] [1, 32, 128, 28] : tensor<1x32x128x128xf16, {order = #NHWC}> to tensor<1x32x128x28xf16, {order = #NHWC}>
        // CHECK:       [[COPY4:%.+]] = VPU.Copy([[SLICE4]])
        // CHECK-SAME{LITERAL}: {out_mem_space = @CMX_NN} : tensor<1x32x128x28xf16, {order = #NHWC}> -> !VPU.DistributedTensor<1x32x128x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 65, 28], [1, 32, 65, 28]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]], memory_shapes = [[1, 32, 65, 28], [1, 32, 65, 28]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}>
        // CHECK:       [[SLICE5:%.+]] = VPU.Slice [[SHAPECAST1]] [0, 0, 0, 51] [1, 64, 128, 28] : tensor<1x64x128x128xf16, {order = #NHWC}> to tensor<1x64x128x28xf16, {order = #NHWC}>
        // CHECK:       [[COPY5:%.+]] = VPU.Copy([[SLICE5]])
        // CHECK-SAME{LITERAL}: {out_mem_space = @CMX_NN} : tensor<1x64x128x28xf16, {order = #NHWC}> -> !VPU.DistributedTensor<1x64x128x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 64, 65, 28], [1, 64, 65, 28]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]], memory_shapes = [[1, 64, 65, 28], [1, 64, 65, 28]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}>
        // CHECK:       [[SLICE6:%.+]] = VPU.Slice [[CONCAT1]] [0, 0, 0, 77] [1, 32, 128, 27] : tensor<1x32x128x128xf16, {order = #NHWC}> to tensor<1x32x128x27xf16, {order = #NHWC}>
        // CHECK:       [[COPY6:%.+]] = VPU.Copy([[SLICE6]])
        // CHECK-SAME{LITERAL}: {out_mem_space = @CMX_NN} : tensor<1x32x128x27xf16, {order = #NHWC}> -> !VPU.DistributedTensor<1x32x128x27xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 65, 27], [1, 32, 65, 27]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]], memory_shapes = [[1, 32, 65, 27], [1, 32, 65, 27]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}>
        // CHECK:       [[SLICE7:%.+]] = VPU.Slice [[SHAPECAST1]] [0, 0, 0, 77] [1, 64, 128, 27] : tensor<1x64x128x128xf16, {order = #NHWC}> to tensor<1x64x128x27xf16, {order = #NHWC}>
        // CHECK:       [[COPY7:%.+]] = VPU.Copy([[SLICE7]])
        // CHECK-SAME{LITERAL}: {out_mem_space = @CMX_NN} : tensor<1x64x128x27xf16, {order = #NHWC}> -> !VPU.DistributedTensor<1x64x128x27xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 64, 65, 27], [1, 64, 65, 27]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]], memory_shapes = [[1, 64, 65, 27], [1, 64, 65, 27]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}>
        // CHECK:       [[SLICE8:%.+]] = VPU.Slice [[CONCAT1]] [0, 0, 0, 102] [1, 32, 128, 26] : tensor<1x32x128x128xf16, {order = #NHWC}> to tensor<1x32x128x26xf16, {order = #NHWC}>
        // CHECK:       [[COPY8:%.+]] = VPU.Copy([[SLICE8]])
        // CHECK-SAME{LITERAL}: {out_mem_space = @CMX_NN} : tensor<1x32x128x26xf16, {order = #NHWC}> -> !VPU.DistributedTensor<1x32x128x26xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 65, 26], [1, 32, 65, 26]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]], memory_shapes = [[1, 32, 65, 26], [1, 32, 65, 26]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}>
        // CHECK:       [[SLICE9:%.+]] = VPU.Slice [[SHAPECAST1]] [0, 0, 0, 102] [1, 64, 128, 26] : tensor<1x64x128x128xf16, {order = #NHWC}> to tensor<1x64x128x26xf16, {order = #NHWC}>
        // CHECK:       [[COPY9:%.+]] = VPU.Copy([[SLICE9]])
        // CHECK-SAME{LITERAL}: {out_mem_space = @CMX_NN} : tensor<1x64x128x26xf16, {order = #NHWC}> -> !VPU.DistributedTensor<1x64x128x26xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 64, 65, 26], [1, 64, 65, 26]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]], memory_shapes = [[1, 64, 65, 26], [1, 64, 65, 26]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}>
        // CHECK:       [[CONCAT2:%.+]] = VPU.Concat([[COPY9]], [[COPY8]])
        // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : !VPU.DistributedTensor<1x64x128x26xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 64, 65, 26], [1, 64, 65, 26]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]], memory_shapes = [[1, 64, 65, 26], [1, 64, 65, 26]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}>, !VPU.DistributedTensor<1x32x128x26xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 65, 26], [1, 32, 65, 26]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]], memory_shapes = [[1, 32, 65, 26], [1, 32, 65, 26]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}> -> !VPU.DistributedTensor<1x96x128x26xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 96, 64, 26], [1, 96, 64, 26]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]], memory_shapes = [[1, 96, 65, 26], [1, 96, 65, 26]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}>
        // CHECK:       [[CONCAT3:%.+]] = VPU.Concat([[COPY7]], [[COPY6]])
        // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : !VPU.DistributedTensor<1x64x128x27xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 64, 65, 27], [1, 64, 65, 27]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]], memory_shapes = [[1, 64, 65, 27], [1, 64, 65, 27]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}>, !VPU.DistributedTensor<1x32x128x27xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 65, 27], [1, 32, 65, 27]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]], memory_shapes = [[1, 32, 65, 27], [1, 32, 65, 27]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}> -> !VPU.DistributedTensor<1x96x128x27xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 96, 64, 27], [1, 96, 64, 27]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]], memory_shapes = [[1, 96, 65, 27], [1, 96, 65, 27]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}>
        // CHECK:       [[CONCAT4:%.+]] = VPU.Concat([[COPY5]], [[COPY4]])
        // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : !VPU.DistributedTensor<1x64x128x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 64, 65, 28], [1, 64, 65, 28]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]], memory_shapes = [[1, 64, 65, 28], [1, 64, 65, 28]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}>, !VPU.DistributedTensor<1x32x128x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 65, 28], [1, 32, 65, 28]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]], memory_shapes = [[1, 32, 65, 28], [1, 32, 65, 28]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}> -> !VPU.DistributedTensor<1x96x128x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 96, 64, 28], [1, 96, 64, 28]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]], memory_shapes = [[1, 96, 65, 28], [1, 96, 65, 28]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}>
        // CHECK:       [[CONCAT5:%.+]] = VPU.Concat([[COPY3]], [[COPY2]])
        // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : !VPU.DistributedTensor<1x64x128x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 64, 65, 28], [1, 64, 65, 28]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]], memory_shapes = [[1, 64, 65, 28], [1, 64, 65, 28]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}>, !VPU.DistributedTensor<1x32x128x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 65, 28], [1, 32, 65, 28]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]], memory_shapes = [[1, 32, 65, 28], [1, 32, 65, 28]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}> -> !VPU.DistributedTensor<1x96x128x28xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 96, 64, 28], [1, 96, 64, 28]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]], memory_shapes = [[1, 96, 65, 28], [1, 96, 65, 28]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}>
        // CHECK:       [[CONCAT6:%.+]] = VPU.Concat([[COPY1]], [[COPY0]])
        // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : !VPU.DistributedTensor<1x64x128x27xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 64, 65, 27], [1, 64, 65, 27]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]], memory_shapes = [[1, 64, 65, 27], [1, 64, 65, 27]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}>, !VPU.DistributedTensor<1x32x128x27xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 32, 65, 27], [1, 32, 65, 27]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]], memory_shapes = [[1, 32, 65, 27], [1, 32, 65, 27]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}> -> !VPU.DistributedTensor<1x96x128x27xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, uniform_distributed_segments, compute_shapes = [[1, 96, 64, 27], [1, 96, 64, 27]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]], memory_shapes = [[1, 96, 65, 27], [1, 96, 65, 27]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]}>

        // CHECK:       return {{%.+}} : tensor<1x32x128x128xf32>
  }
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0014466386799718818:108>

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>

module @EnableWeightDeqauntEnsuranceBeforeStrategy {
  config.Resources 2 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1326182 bytes of @CMX_NN_FragmentationAware
    config.MemoryResource 1473536 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
  }
  config.Resources 1 of @global {
    config.ExecutorResource 1 of @M2I
    config.ExecutorResource 2 of @DMA_NN
    config.MemoryResource 67108864000 bytes of @DDR {config.bandwidth = 64 : i64, config.derateFactor = 6.000000e-01 : f64}
  }
  net.NetworkInfo entryPoint : @main inputsInfo : {
    DataInfo "input" tensorNames = ["input"] : tensor<1x128x256x1xf16>
  } outputsInfo : {
    DataInfo "output" friendlyName = "output" : tensor<1x10240x64x4xf16>
  }
    // CHECK-LABEL: func.func @main
    // CHECK-SAME:        [[ARG_0:%[^:]+]]: tensor<1x128x256x1xf16, {order = #NHWC}>
    func.func @main(%arg0: tensor<1x128x256x1xf16, {order = #NHWC}>) -> tensor<1x10240x64x4xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<10240x128x1x1x!qElemType, {order = #NHWC}> = dense<10> : tensor<128x10240xui8>, [#const.Reshape<[1, 1, 128, 10240]>, #const.CastElemType<f16>, #const.CastElemType<!qElemType>, #const.AffineReshape<[[0], [0], [1, 2], [3]], [1, 128, 1, 10240]>, #const.Transpose<#NWHC>, #const.AffineReshape<[[0], [0], [0], [1, 2, 3]], [10240, 128, 1, 1]>, #const.MemPermute<#NHWC, #NHWC>]
    %cst_0 = const.Declare tensor<10240x1x1x4xsi32> = dense<10> : tensor<10240x1x1x4xsi32>
    %0 = VPU.Dequantize(%cst) {dstElemType = f16} : tensor<10240x128x1x1x!qElemType, {order = #NHWC}> -> tensor<10240x128x1x1xf16, {order = #NHWC}>
    %1 = VPU.AffineReshape(%arg0) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 128, 64, 4]} : tensor<1x128x256x1xf16, {order = #NHWC}> -> tensor<1x128x64x4xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %0, %cst_0) {mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, prelu_alpha = [1.000000e+00], adder = 0.000000e+00 : f64>, rawFilterShape = [10240, 128, 1, 1], strides = [1, 1]} : tensor<1x128x64x4xf16, {order = #NHWC}>, tensor<10240x128x1x1xf16, {order = #NHWC}>, tensor<10240x1x1x4xsi32> -> tensor<1x10240x64x4xf16, {order = #NHWC}>
   return %2 : tensor<1x10240x64x4xf16, {order = #NHWC}>


    // CHECK-DAG:   [[WEIGHTS0:%.+]] = const.Declare tensor<640x128x1x1x!qElemType, {order = #NHWC}> = dense<10> : tensor<128x10240xui8>, [#const.SubView<[0, 0], [128, 640]>
    // CHECK-DAG:   [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<640x1x1x4xsi32> = dense<10> : tensor<10240x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [640, 1, 1, 4]>]
    // CHECK-DAG:   [[WEIGHTS1:%.+]] = const.Declare tensor<640x128x1x1x!qElemType, {order = #NHWC}> = dense<10> : tensor<128x10240xui8>, [#const.SubView<[0, 640], [128, 640]>
    // CHECK-DAG:   [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<640x1x1x4xsi32> = dense<10> : tensor<10240x1x1x4xsi32>, [#const.SubView<[640, 0, 0, 0], [640, 1, 1, 4]>]
    // CHECK-DAG:   [[WEIGHTS2:%.+]] = const.Declare tensor<640x128x1x1x!qElemType, {order = #NHWC}> = dense<10> : tensor<128x10240xui8>, [#const.SubView<[0, 1280], [128, 640]>
    // CHECK-DAG:   [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<640x1x1x4xsi32> = dense<10> : tensor<10240x1x1x4xsi32>, [#const.SubView<[1280, 0, 0, 0], [640, 1, 1, 4]>]
    // CHECK-DAG:   [[WEIGHTS3:%.+]] = const.Declare tensor<640x128x1x1x!qElemType, {order = #NHWC}> = dense<10> : tensor<128x10240xui8>, [#const.SubView<[0, 1920], [128, 640]>
    // CHECK-DAG:   [[WEIGHTS_TABLE3:%.+]] = const.Declare tensor<640x1x1x4xsi32> = dense<10> : tensor<10240x1x1x4xsi32>, [#const.SubView<[1920, 0, 0, 0], [640, 1, 1, 4]>]
    // CHECK-DAG:   [[WEIGHTS4:%.+]] = const.Declare tensor<640x128x1x1x!qElemType, {order = #NHWC}> = dense<10> : tensor<128x10240xui8>, [#const.SubView<[0, 2560], [128, 640]>
    // CHECK-DAG:   [[WEIGHTS_TABLE4:%.+]] = const.Declare tensor<640x1x1x4xsi32> = dense<10> : tensor<10240x1x1x4xsi32>, [#const.SubView<[2560, 0, 0, 0], [640, 1, 1, 4]>]
    // CHECK-DAG:   [[WEIGHTS5:%.+]] = const.Declare tensor<640x128x1x1x!qElemType, {order = #NHWC}> = dense<10> : tensor<128x10240xui8>, [#const.SubView<[0, 3200], [128, 640]>
    // CHECK-DAG:   [[WEIGHTS_TABLE5:%.+]] = const.Declare tensor<640x1x1x4xsi32> = dense<10> : tensor<10240x1x1x4xsi32>, [#const.SubView<[3200, 0, 0, 0], [640, 1, 1, 4]>]
    // CHECK-DAG:   [[WEIGHTS6:%.+]] = const.Declare tensor<640x128x1x1x!qElemType, {order = #NHWC}> = dense<10> : tensor<128x10240xui8>, [#const.SubView<[0, 3840], [128, 640]>
    // CHECK-DAG:   [[WEIGHTS_TABLE6:%.+]] = const.Declare tensor<640x1x1x4xsi32> = dense<10> : tensor<10240x1x1x4xsi32>, [#const.SubView<[3840, 0, 0, 0], [640, 1, 1, 4]>]
    // CHECK-DAG:   [[WEIGHTS7:%.+]] = const.Declare tensor<640x128x1x1x!qElemType, {order = #NHWC}> = dense<10> : tensor<128x10240xui8>, [#const.SubView<[0, 4480], [128, 640]>
    // CHECK-DAG:   [[WEIGHTS_TABLE7:%.+]] = const.Declare tensor<640x1x1x4xsi32> = dense<10> : tensor<10240x1x1x4xsi32>, [#const.SubView<[4480, 0, 0, 0], [640, 1, 1, 4]>]
    // CHECK-DAG:   [[WEIGHTS8:%.+]] = const.Declare tensor<640x128x1x1x!qElemType, {order = #NHWC}> = dense<10> : tensor<128x10240xui8>, [#const.SubView<[0, 5120], [128, 640]>
    // CHECK-DAG:   [[WEIGHTS_TABLE8:%.+]] = const.Declare tensor<640x1x1x4xsi32> = dense<10> : tensor<10240x1x1x4xsi32>, [#const.SubView<[5120, 0, 0, 0], [640, 1, 1, 4]>]
    // CHECK-DAG:   [[WEIGHTS9:%.+]] = const.Declare tensor<640x128x1x1x!qElemType, {order = #NHWC}> = dense<10> : tensor<128x10240xui8>, [#const.SubView<[0, 5760], [128, 640]>
    // CHECK-DAG:   [[WEIGHTS_TABLE9:%.+]] = const.Declare tensor<640x1x1x4xsi32> = dense<10> : tensor<10240x1x1x4xsi32>, [#const.SubView<[5760, 0, 0, 0], [640, 1, 1, 4]>]
    // CHECK-DAG:   [[WEIGHTS10:%.+]] = const.Declare tensor<640x128x1x1x!qElemType, {order = #NHWC}> = dense<10> : tensor<128x10240xui8>, [#const.SubView<[0, 6400], [128, 640]>
    // CHECK-DAG:   [[WEIGHTS_TABLE10:%.+]] = const.Declare tensor<640x1x1x4xsi32> = dense<10> : tensor<10240x1x1x4xsi32>, [#const.SubView<[6400, 0, 0, 0], [640, 1, 1, 4]>]
    // CHECK-DAG:   [[WEIGHTS11:%.+]] = const.Declare tensor<640x128x1x1x!qElemType, {order = #NHWC}> = dense<10> : tensor<128x10240xui8>, [#const.SubView<[0, 7040], [128, 640]>
    // CHECK-DAG:   [[WEIGHTS_TABLE11:%.+]] = const.Declare tensor<640x1x1x4xsi32> = dense<10> : tensor<10240x1x1x4xsi32>, [#const.SubView<[7040, 0, 0, 0], [640, 1, 1, 4]>]
    // CHECK-DAG:   [[WEIGHTS12:%.+]] = const.Declare tensor<640x128x1x1x!qElemType, {order = #NHWC}> = dense<10> : tensor<128x10240xui8>, [#const.SubView<[0, 7680], [128, 640]>
    // CHECK-DAG:   [[WEIGHTS_TABLE12:%.+]] = const.Declare tensor<640x1x1x4xsi32> = dense<10> : tensor<10240x1x1x4xsi32>, [#const.SubView<[7680, 0, 0, 0], [640, 1, 1, 4]>]
    // CHECK-DAG:   [[WEIGHTS13:%.+]] = const.Declare tensor<640x128x1x1x!qElemType, {order = #NHWC}> = dense<10> : tensor<128x10240xui8>, [#const.SubView<[0, 8320], [128, 640]>
    // CHECK-DAG:   [[WEIGHTS_TABLE13:%.+]] = const.Declare tensor<640x1x1x4xsi32> = dense<10> : tensor<10240x1x1x4xsi32>, [#const.SubView<[8320, 0, 0, 0], [640, 1, 1, 4]>]
    // CHECK-DAG:   [[WEIGHTS14:%.+]] = const.Declare tensor<640x128x1x1x!qElemType, {order = #NHWC}> = dense<10> : tensor<128x10240xui8>, [#const.SubView<[0, 8960], [128, 640]>
    // CHECK-DAG:   [[WEIGHTS_TABLE14:%.+]] = const.Declare tensor<640x1x1x4xsi32> = dense<10> : tensor<10240x1x1x4xsi32>, [#const.SubView<[8960, 0, 0, 0], [640, 1, 1, 4]>]
    // CHECK-DAG:   [[WEIGHTS15:%.+]] = const.Declare tensor<640x128x1x1x!qElemType, {order = #NHWC}> = dense<10> : tensor<128x10240xui8>, [#const.SubView<[0, 9600], [128, 640]>
    // CHECK-DAG:   [[WEIGHTS_TABLE15:%.+]] = const.Declare tensor<640x1x1x4xsi32> = dense<10> : tensor<10240x1x1x4xsi32>, [#const.SubView<[9600, 0, 0, 0], [640, 1, 1, 4]>]

    // CHECK: [[RESHAPE:%.+]] = VPU.AffineReshape([[ARG_0]])
    // CHECK-SAME{LITERAL}:         {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 128, 64, 4]}

    // CHECK: [[COPY_W0:%.+]] = VPU.Copy([[WEIGHTS0]]) {out_mem_space = @CMX_NN}
    // CHECK: [[DEQ0:%.+]] = VPU.Dequantize([[COPY_W0]]) {dstElemType = f16, vf_loop_index = 0 : i64, vf_loop_layer_index = 0 : i64}
    // CHECK: [[COPY_IN0:%.+]] = VPU.Copy([[RESHAPE]]) {out_mem_space = @CMX_NN}
    // CHECK: [[COPY_T0:%.+]] = VPU.Copy([[WEIGHTS_TABLE0]]) {out_mem_space = @CMX_NN}
    // CHECK: [[CONV0:%.+]] = VPU.NCE.Convolution([[COPY_IN0]], [[DEQ0]], [[COPY_T0]])
    // CHECK: [[OUT0:%.+]] = VPU.Copy([[CONV0]])

    // CHECK: [[COPY_W1:%.+]] = VPU.Copy([[WEIGHTS1]]) {out_mem_space = @CMX_NN}
    // CHECK: [[DEQ1:%.+]] = VPU.Dequantize([[COPY_W1]]) {dstElemType = f16, vf_loop_index = 0 : i64, vf_loop_layer_index = 1 : i64}
    // CHECK: [[COPY_IN1:%.+]] = VPU.Copy([[RESHAPE]]) {out_mem_space = @CMX_NN}
    // CHECK: [[COPY_T1:%.+]] = VPU.Copy([[WEIGHTS_TABLE1]]) {out_mem_space = @CMX_NN}
    // CHECK: [[CONV1:%.+]] = VPU.NCE.Convolution([[COPY_IN1]], [[DEQ1]], [[COPY_T1]])
    // CHECK: [[OUT1:%.+]] = VPU.Copy([[CONV1]])

    // CHECK: [[COPY_W2:%.+]] = VPU.Copy([[WEIGHTS2]]) {out_mem_space = @CMX_NN}
    // CHECK: [[DEQ2:%.+]] = VPU.Dequantize([[COPY_W2]]) {dstElemType = f16, vf_loop_index = 0 : i64, vf_loop_layer_index = 2 : i64}
    // CHECK: [[COPY_IN2:%.+]] = VPU.Copy([[RESHAPE]]) {out_mem_space = @CMX_NN}
    // CHECK: [[COPY_T2:%.+]] = VPU.Copy([[WEIGHTS_TABLE2]]) {out_mem_space = @CMX_NN}
    // CHECK: [[CONV2:%.+]] = VPU.NCE.Convolution([[COPY_IN2]], [[DEQ2]], [[COPY_T2]])
    // CHECK: [[OUT2:%.+]] = VPU.Copy([[CONV2]])

    // CHECK: [[COPY_W3:%.+]] = VPU.Copy([[WEIGHTS3]]) {out_mem_space = @CMX_NN}
    // CHECK: [[DEQ3:%.+]] = VPU.Dequantize([[COPY_W3]]) {dstElemType = f16, vf_loop_index = 0 : i64, vf_loop_layer_index = 3 : i64}
    // CHECK: [[COPY_IN3:%.+]] = VPU.Copy([[RESHAPE]]) {out_mem_space = @CMX_NN}
    // CHECK: [[COPY_T3:%.+]] = VPU.Copy([[WEIGHTS_TABLE3]]) {out_mem_space = @CMX_NN}
    // CHECK: [[CONV3:%.+]] = VPU.NCE.Convolution([[COPY_IN3]], [[DEQ3]], [[COPY_T3]])
    // CHECK: [[OUT3:%.+]] = VPU.Copy([[CONV3]])

    // CHECK: [[COPY_W4:%.+]] = VPU.Copy([[WEIGHTS4]]) {out_mem_space = @CMX_NN}
    // CHECK: [[DEQ4:%.+]] = VPU.Dequantize([[COPY_W4]]) {dstElemType = f16, vf_loop_index = 0 : i64, vf_loop_layer_index = 4 : i64}
    // CHECK: [[COPY_IN4:%.+]] = VPU.Copy([[RESHAPE]]) {out_mem_space = @CMX_NN}
    // CHECK: [[COPY_T4:%.+]] = VPU.Copy([[WEIGHTS_TABLE4]]) {out_mem_space = @CMX_NN}
    // CHECK: [[CONV4:%.+]] = VPU.NCE.Convolution([[COPY_IN4]], [[DEQ4]], [[COPY_T4]])
    // CHECK: [[OUT4:%.+]] = VPU.Copy([[CONV4]])

    // CHECK: [[COPY_W5:%.+]] = VPU.Copy([[WEIGHTS5]]) {out_mem_space = @CMX_NN}
    // CHECK: [[DEQ5:%.+]] = VPU.Dequantize([[COPY_W5]]) {dstElemType = f16, vf_loop_index = 0 : i64, vf_loop_layer_index = 5 : i64}
    // CHECK: [[COPY_IN5:%.+]] = VPU.Copy([[RESHAPE]]) {out_mem_space = @CMX_NN}
    // CHECK: [[COPY_T5:%.+]] = VPU.Copy([[WEIGHTS_TABLE5]]) {out_mem_space = @CMX_NN}
    // CHECK: [[CONV5:%.+]] = VPU.NCE.Convolution([[COPY_IN5]], [[DEQ5]], [[COPY_T5]])
    // CHECK: [[OUT5:%.+]] = VPU.Copy([[CONV5]])

    // CHECK: [[COPY_W6:%.+]] = VPU.Copy([[WEIGHTS6]]) {out_mem_space = @CMX_NN}
    // CHECK: [[DEQ6:%.+]] = VPU.Dequantize([[COPY_W6]]) {dstElemType = f16, vf_loop_index = 0 : i64, vf_loop_layer_index = 6 : i64}
    // CHECK: [[COPY_IN6:%.+]] = VPU.Copy([[RESHAPE]]) {out_mem_space = @CMX_NN}
    // CHECK: [[COPY_T6:%.+]] = VPU.Copy([[WEIGHTS_TABLE6]]) {out_mem_space = @CMX_NN}
    // CHECK: [[CONV6:%.+]] = VPU.NCE.Convolution([[COPY_IN6]], [[DEQ6]], [[COPY_T6]])
    // CHECK: [[OUT6:%.+]] = VPU.Copy([[CONV6]])

    // CHECK: [[COPY_W7:%.+]] = VPU.Copy([[WEIGHTS7]]) {out_mem_space = @CMX_NN}
    // CHECK: [[DEQ7:%.+]] = VPU.Dequantize([[COPY_W7]]) {dstElemType = f16, vf_loop_index = 0 : i64, vf_loop_layer_index = 7 : i64}
    // CHECK: [[COPY_IN7:%.+]] = VPU.Copy([[RESHAPE]]) {out_mem_space = @CMX_NN}
    // CHECK: [[COPY_T7:%.+]] = VPU.Copy([[WEIGHTS_TABLE7]]) {out_mem_space = @CMX_NN}
    // CHECK: [[CONV7:%.+]] = VPU.NCE.Convolution([[COPY_IN7]], [[DEQ7]], [[COPY_T7]])
    // CHECK: [[OUT7:%.+]] = VPU.Copy([[CONV7]])

    // CHECK: [[COPY_W8:%.+]] = VPU.Copy([[WEIGHTS8]]) {out_mem_space = @CMX_NN}
    // CHECK: [[DEQ8:%.+]] = VPU.Dequantize([[COPY_W8]]) {dstElemType = f16, vf_loop_index = 0 : i64, vf_loop_layer_index = 8 : i64}
    // CHECK: [[COPY_IN8:%.+]] = VPU.Copy([[RESHAPE]]) {out_mem_space = @CMX_NN}
    // CHECK: [[COPY_T8:%.+]] = VPU.Copy([[WEIGHTS_TABLE8]]) {out_mem_space = @CMX_NN}
    // CHECK: [[CONV8:%.+]] = VPU.NCE.Convolution([[COPY_IN8]], [[DEQ8]], [[COPY_T8]])
    // CHECK: [[OUT8:%.+]] = VPU.Copy([[CONV8]])

    // CHECK: [[COPY_W9:%.+]] = VPU.Copy([[WEIGHTS9]]) {out_mem_space = @CMX_NN}
    // CHECK: [[DEQ9:%.+]] = VPU.Dequantize([[COPY_W9]]) {dstElemType = f16, vf_loop_index = 0 : i64, vf_loop_layer_index = 9 : i64}
    // CHECK: [[COPY_IN9:%.+]] = VPU.Copy([[RESHAPE]]) {out_mem_space = @CMX_NN}
    // CHECK: [[COPY_T9:%.+]] = VPU.Copy([[WEIGHTS_TABLE9]]) {out_mem_space = @CMX_NN}
    // CHECK: [[CONV9:%.+]] = VPU.NCE.Convolution([[COPY_IN9]], [[DEQ9]], [[COPY_T9]])
    // CHECK: [[OUT9:%.+]] = VPU.Copy([[CONV9]])

    // CHECK: [[COPY_W10:%.+]] = VPU.Copy([[WEIGHTS10]]) {out_mem_space = @CMX_NN}
    // CHECK: [[DEQ10:%.+]] = VPU.Dequantize([[COPY_W10]]) {dstElemType = f16, vf_loop_index = 0 : i64, vf_loop_layer_index = 10 : i64}
    // CHECK: [[COPY_IN10:%.+]] = VPU.Copy([[RESHAPE]]) {out_mem_space = @CMX_NN}
    // CHECK: [[COPY_T10:%.+]] = VPU.Copy([[WEIGHTS_TABLE10]]) {out_mem_space = @CMX_NN}
    // CHECK: [[CONV10:%.+]] = VPU.NCE.Convolution([[COPY_IN10]], [[DEQ10]], [[COPY_T10]])
    // CHECK: [[OUT10:%.+]] = VPU.Copy([[CONV10]])

    // CHECK: [[COPY_W11:%.+]] = VPU.Copy([[WEIGHTS11]]) {out_mem_space = @CMX_NN}
    // CHECK: [[DEQ11:%.+]] = VPU.Dequantize([[COPY_W11]]) {dstElemType = f16, vf_loop_index = 0 : i64, vf_loop_layer_index = 11 : i64}
    // CHECK: [[COPY_IN11:%.+]] = VPU.Copy([[RESHAPE]]) {out_mem_space = @CMX_NN}
    // CHECK: [[COPY_T11:%.+]] = VPU.Copy([[WEIGHTS_TABLE11]]) {out_mem_space = @CMX_NN}
    // CHECK: [[CONV11:%.+]] = VPU.NCE.Convolution([[COPY_IN11]], [[DEQ11]], [[COPY_T11]])
    // CHECK: [[OUT11:%.+]] = VPU.Copy([[CONV11]])

    // CHECK: [[COPY_W12:%.+]] = VPU.Copy([[WEIGHTS12]]) {out_mem_space = @CMX_NN}
    // CHECK: [[DEQ12:%.+]] = VPU.Dequantize([[COPY_W12]]) {dstElemType = f16, vf_loop_index = 0 : i64, vf_loop_layer_index = 12 : i64}
    // CHECK: [[COPY_IN12:%.+]] = VPU.Copy([[RESHAPE]]) {out_mem_space = @CMX_NN}
    // CHECK: [[COPY_T12:%.+]] = VPU.Copy([[WEIGHTS_TABLE12]]) {out_mem_space = @CMX_NN}
    // CHECK: [[CONV12:%.+]] = VPU.NCE.Convolution([[COPY_IN12]], [[DEQ12]], [[COPY_T12]])
    // CHECK: [[OUT12:%.+]] = VPU.Copy([[CONV12]])

    // CHECK: [[COPY_W13:%.+]] = VPU.Copy([[WEIGHTS13]]) {out_mem_space = @CMX_NN}
    // CHECK: [[DEQ13:%.+]] = VPU.Dequantize([[COPY_W13]]) {dstElemType = f16, vf_loop_index = 0 : i64, vf_loop_layer_index = 13 : i64}
    // CHECK: [[COPY_IN13:%.+]] = VPU.Copy([[RESHAPE]]) {out_mem_space = @CMX_NN}
    // CHECK: [[COPY_T13:%.+]] = VPU.Copy([[WEIGHTS_TABLE13]]) {out_mem_space = @CMX_NN}
    // CHECK: [[CONV13:%.+]] = VPU.NCE.Convolution([[COPY_IN13]], [[DEQ13]], [[COPY_T13]])
    // CHECK: [[OUT13:%.+]] = VPU.Copy([[CONV13]])

    // CHECK: [[COPY_W14:%.+]] = VPU.Copy([[WEIGHTS14]]) {out_mem_space = @CMX_NN}
    // CHECK: [[DEQ14:%.+]] = VPU.Dequantize([[COPY_W14]]) {dstElemType = f16, vf_loop_index = 0 : i64, vf_loop_layer_index = 14 : i64}
    // CHECK: [[COPY_IN14:%.+]] = VPU.Copy([[RESHAPE]]) {out_mem_space = @CMX_NN}
    // CHECK: [[COPY_T14:%.+]] = VPU.Copy([[WEIGHTS_TABLE14]]) {out_mem_space = @CMX_NN}
    // CHECK: [[CONV14:%.+]] = VPU.NCE.Convolution([[COPY_IN14]], [[DEQ14]], [[COPY_T14]])
    // CHECK: [[OUT14:%.+]] = VPU.Copy([[CONV14]])

    // CHECK: [[COPY_W15:%.+]] = VPU.Copy([[WEIGHTS15]]) {out_mem_space = @CMX_NN}
    // CHECK: [[DEQ15:%.+]] = VPU.Dequantize([[COPY_W15]]) {dstElemType = f16, vf_loop_index = 0 : i64, vf_loop_layer_index = 15 : i64}
    // CHECK: [[COPY_IN15:%.+]] = VPU.Copy([[RESHAPE]]) {out_mem_space = @CMX_NN}
    // CHECK: [[COPY_T15:%.+]] = VPU.Copy([[WEIGHTS_TABLE15]]) {out_mem_space = @CMX_NN}
    // CHECK: [[CONV15:%.+]] = VPU.NCE.Convolution([[COPY_IN15]], [[DEQ15]], [[COPY_T15]])
    // CHECK: [[OUT15:%.+]] = VPU.Copy([[CONV15]])

    // CHECK: [[CONCAT:%.+]] = VPU.Concat([[OUT0]], [[OUT1]], [[OUT2]], [[OUT3]], [[OUT4]], [[OUT5]], [[OUT6]], [[OUT7]], [[OUT8]], [[OUT9]], [[OUT10]], [[OUT11]], [[OUT12]], [[OUT13]], [[OUT14]], [[OUT15]])
    // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [0, 640, 0, 0], [0, 1280, 0, 0], [0, 1920, 0, 0], [0, 2560, 0, 0], [0, 3200, 0, 0], [0, 3840, 0, 0], [0, 4480, 0, 0], [0, 5120, 0, 0], [0, 5760, 0, 0], [0, 6400, 0, 0], [0, 7040, 0, 0], [0, 7680, 0, 0], [0, 8320, 0, 0], [0, 8960, 0, 0], [0, 9600, 0, 0]]}
    // CHECK: return [[CONCAT]]
}
}
