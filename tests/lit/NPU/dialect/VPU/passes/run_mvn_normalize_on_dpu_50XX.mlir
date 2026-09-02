//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --run-mvn-normalize-on-dpu %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// MVN1NormalizeOp with from_force_decompose = true: came from the forceDecompose path in
// DecomposeMVNPass (previously a CMX-fitting monolithic-Shave MVN op). This is the path that
// benefits from DPU conversion: Shave is freed, DPU does the normalize.
// Verify it IS converted to NCEMaxPool.
// CHECK-LABEL: func.func @ReplaceMVN1NormalizeWithMaxPool
// CHECK-SAME:        [[INPUT:%[^:]+]]: tensor<1x512x64x64xf32>
func.func @ReplaceMVN1NormalizeWithMaxPool(%arg0: tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32> {
    %0 = VPU.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x512x64x64xf32> -> tensor<1x64x512x64xf32, {order = #NHWC}>
    %1 = VPU.ShapeCast {shape = [1, 512, 64, 64]} inputs(%0 : tensor<1x64x512x64xf32, {order = #NHWC}>) -> tensor<1x512x64x64xf32, {order = #NHWC}>
    %2 = VPU.Convert(%1) {dstElemType = f16} : tensor<1x512x64x64xf32, {order = #NHWC}> -> tensor<1x512x64x64xf16, {order = #NHWC}>
    %3 = VPU.ShapeCast {shape = [1, 512, 4096, 1]} inputs(%2 : tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x512x4096x1xf16, {order = #NHWC}>
    %4 = VPU.MVN1SumOp(%3) {across_channels = false, normalize_variance = true, output_height = 3 : i64} : tensor<1x512x4096x1xf16, {order = #NHWC}> -> tensor<1x512x3x2xf32, {order = #NHWC}>
    %5 = VPU.MVN1MeanVar(%4) {across_channels = false, eps = 9.9999999999999995E-7 : f64, internal_reshape = [1, 32, 16, 4096], normalize_variance = true, orig_shape = [1, 512, 64, 64], output_type = f16} : tensor<1x512x3x2xf32, {order = #NHWC}> -> tensor<1x512x1x2xf16, {order = #NHWC}>
    %6 = VPU.MVN1Normalize(%3, %5) {across_channels = false, normalize_variance = true, from_force_decompose = true} : tensor<1x512x4096x1xf16, {order = #NHWC}>, tensor<1x512x1x2xf16, {order = #NHWC}> -> tensor<1x512x4096x1xf16, {order = #NHWC}>
    %7 = VPU.ShapeCast {shape = [1, 512, 64, 64]} inputs(%6 : tensor<1x512x4096x1xf16, {order = #NHWC}>) -> tensor<1x512x64x64xf16, {order = #NHWC}>
    %8 = VPU.Convert(%7) {dstElemType = f32} : tensor<1x512x64x64xf16, {order = #NHWC}> -> tensor<1x512x64x64xf32, {order = #NHWC}>
    %9 = VPU.ShapeCast {shape = [1, 64, 512, 64]} inputs(%8 : tensor<1x512x64x64xf32, {order = #NHWC}>) -> tensor<1x64x512x64xf32, {order = #NHWC}>
    %10 = VPU.PermuteCast(%9) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x512x64xf32, {order = #NHWC}> -> tensor<1x512x64x64xf32>
    return %10 : tensor<1x512x64x64xf32>

    // CHECK:        [[PERMUTECAST_1:%.+]] = VPU.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCHW}
    // CHECK-SAME:       : tensor<1x512x64x64xf32> -> tensor<1x64x512x64xf32, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_1:%.+]] = VPU.ShapeCast {shape = [1, 512, 64, 64]} inputs([[PERMUTECAST_1]] : tensor<1x64x512x64xf32, {order = #NHWC}>)
    // CHECK-SAME:       -> tensor<1x512x64x64xf32, {order = #NHWC}>

    // CHECK:        [[CONVERT_1:%.+]] = VPU.Convert([[SHAPECAST_1]]) {dstElemType = f16}
    // CHECK-SAME:       : tensor<1x512x64x64xf32, {order = #NHWC}> -> tensor<1x512x64x64xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_2:%.+]] = VPU.ShapeCast {shape = [1, 512, 4096, 1]} inputs([[CONVERT_1]] : tensor<1x512x64x64xf16, {order = #NHWC}>)
    // CHECK-SAME:       -> tensor<1x512x4096x1xf16, {order = #NHWC}>

    // CHECK:        [[SUM:%.+]] = VPU.MVN1SumOp([[SHAPECAST_2]]) {across_channels = false, normalize_variance = true, output_height = 3 : i64}
    // CHECK-SAME:       : tensor<1x512x4096x1xf16, {order = #NHWC}> -> tensor<1x512x3x2xf32, {order = #NHWC}>

    // CHECK:        [[MEANVAR:%.+]] = VPU.MVN1MeanVar([[SUM]])
    // CHECK-SAME:       {across_channels = false, eps = 9.9999999999999995E-7 : f64, internal_reshape = [1, 32, 16, 4096], normalize_variance = true, orig_shape = [1, 512, 64, 64], output_type = f16}
    // CHECK-SAME:       : tensor<1x512x3x2xf32, {order = #NHWC}> -> tensor<1x512x1x2xf16, {order = #NHWC}>

    // CHECK:        [[SLICE_MEAN:%.+]] = VPU.Slice [[MEANVAR]] [0, 0, 0, 0] [1, 512, 1, 1]
    // CHECK-SAME:       : tensor<1x512x1x2xf16, {order = #NHWC}> to tensor<1x512x1x1xf16, {order = #NHWC}>

    // CHECK:        [[SLICE_SCALE:%.+]] = VPU.Slice [[MEANVAR]] [0, 0, 0, 1] [1, 512, 1, 1]
    // CHECK-SAME:       : tensor<1x512x1x2xf16, {order = #NHWC}> to tensor<1x512x1x1xf16, {order = #NHWC}>

    // Scale is converted to fp32 via a DPU identity NCE.MaxPool (convertToFp32ViaIdentityMaxPool),
    // reshaping to a near-square [1,64,2,4] (genuine spatial extent) rather than [512,1,1,1].
    // CHECK:        [[SCALE_RESHAPE_IN:%.+]] = VPU.ShapeCast {shape = [1, 64, 2, 4]} inputs([[SLICE_SCALE]] : tensor<1x512x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:       -> tensor<1x64x2x4xf16, {order = #NHWC}>

    // CHECK-DAG:    [[SCALE_ONE:%.+]] = const.Declare tensor<64x1x1x1xf32, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x1x1x1xf32>
    // CHECK-DAG:    [[SCALE_ZERO:%.+]] = const.Declare tensor<64x1x1x1xf32, {order = #NHWC}> = dense<0.000000e+00> : tensor<64x1x1x1xf32>
    // CHECK-DAG:    [[SCALE_WT_ZEROS:%.+]] = const.Declare tensor<64x1x1x2xf32, {order = #NHWC}> = dense<0.000000e+00> : tensor<64x1x1x2xf32>

    // CHECK:        [[SCALE_WT_CONCAT:%.+]] = VPU.Concat([[SCALE_WT_ZEROS]], [[SCALE_ONE]], [[SCALE_ZERO]])

    // CHECK:        [[SCALE_WT:%.+]] = Core.ReinterpretCast([[SCALE_WT_CONCAT]])
    // CHECK-SAME:       -> tensor<64x1x1x4xsi32, {order = #NHWC}>

    // CHECK:        [[SCALE_MAXPOOL:%.+]] = VPU.NCE.MaxPool([[SCALE_RESHAPE_IN]], [[SCALE_WT]]
    // CHECK-SAME:       -> tensor<1x64x2x4xf32, {order = #NHWC}>

    // CHECK:        [[CONVERT_2:%.+]] = VPU.ShapeCast {shape = [512, 1, 1, 1]} inputs([[SCALE_MAXPOOL]] : tensor<1x64x2x4xf32, {order = #NHWC}>)
    // CHECK-SAME:       -> tensor<512x1x1x1xf32, {order = #NHWC}>

    // CHECK-DAG:    [[CST_NEG:%.+]] = const.Declare tensor<1x512x1x1xf16, {order = #NHWC}> = dense<-1.000000e+00> : tensor<1x512x1x1xf16>

    // CHECK:        [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[SLICE_MEAN]], [[CST_NEG]])
    // CHECK-SAME:       op_type = #VPU.eltwise_type<MULTIPLY>

    // Bias (-mean) is likewise converted to fp32 via a DPU identity NCE.MaxPool.
    // CHECK:        [[BIAS_RESHAPE_IN:%.+]] = VPU.ShapeCast {shape = [1, 64, 2, 4]} inputs([[ELTWISE]] : tensor<1x512x1x1xf16, {order = #NHWC}>)
    // CHECK-SAME:       -> tensor<1x64x2x4xf16, {order = #NHWC}>

    // CHECK-DAG:    [[BIAS_ONE:%.+]] = const.Declare tensor<64x1x1x1xf32, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x1x1x1xf32>
    // CHECK-DAG:    [[BIAS_ZERO:%.+]] = const.Declare tensor<64x1x1x1xf32, {order = #NHWC}> = dense<0.000000e+00> : tensor<64x1x1x1xf32>
    // CHECK-DAG:    [[BIAS_WT_ZEROS:%.+]] = const.Declare tensor<64x1x1x2xf32, {order = #NHWC}> = dense<0.000000e+00> : tensor<64x1x1x2xf32>

    // CHECK:        [[BIAS_WT_CONCAT:%.+]] = VPU.Concat([[BIAS_WT_ZEROS]], [[BIAS_ONE]], [[BIAS_ZERO]])

    // CHECK:        [[BIAS_WT:%.+]] = Core.ReinterpretCast([[BIAS_WT_CONCAT]])
    // CHECK-SAME:       -> tensor<64x1x1x4xsi32, {order = #NHWC}>

    // CHECK:        [[BIAS_MAXPOOL:%.+]] = VPU.NCE.MaxPool([[BIAS_RESHAPE_IN]], [[BIAS_WT]]
    // CHECK-SAME:       -> tensor<1x64x2x4xf32, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_4:%.+]] = VPU.ShapeCast {shape = [512, 1, 1, 1]} inputs([[BIAS_MAXPOOL]] : tensor<1x64x2x4xf32, {order = #NHWC}>)
    // CHECK-SAME:       -> tensor<512x1x1x1xf32, {order = #NHWC}>

    // CHECK-DAG:    [[CST_ZEROS:%.+]] = const.Declare tensor<512x1x1x2xf32, {order = #NHWC}> = dense<0.000000e+00> : tensor<512x1x1x2xf32>

    // CHECK:        [[CONCAT:%.+]] = VPU.Concat([[CST_ZEROS]], [[CONVERT_2]], [[SHAPECAST_4]])
    // CHECK-SAME:       {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 3]]}

    // CHECK:        [[REINTERPRET:%.+]] = Core.ReinterpretCast([[CONCAT]])
    // CHECK-SAME:       : tensor<512x1x1x4xf32, {order = #NHWC}> -> tensor<512x1x1x4xsi32, {order = #NHWC}>

    // CHECK:        [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[SHAPECAST_2]], [[REINTERPRET]] )
    // CHECK-SAME:       -> tensor<1x512x4096x1xf16, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_5:%.+]] = VPU.ShapeCast {shape = [1, 512, 64, 64]} inputs([[MAXPOOL]] : tensor<1x512x4096x1xf16, {order = #NHWC}>)
    // CHECK-SAME:       -> tensor<1x512x64x64xf16, {order = #NHWC}>

    // CHECK:        [[CONVERT_4:%.+]] = VPU.Convert([[SHAPECAST_5]]) {dstElemType = f32}
    // CHECK-SAME:       : tensor<1x512x64x64xf16, {order = #NHWC}> -> tensor<1x512x64x64xf32, {order = #NHWC}>

    // CHECK:        [[SHAPECAST_6:%.+]] = VPU.ShapeCast {shape = [1, 64, 512, 64]} inputs([[CONVERT_4]] : tensor<1x512x64x64xf32, {order = #NHWC}>)
    // CHECK-SAME:       -> tensor<1x64x512x64xf32, {order = #NHWC}>

    // CHECK:        [[PERMUTECAST_2:%.+]] = VPU.PermuteCast([[SHAPECAST_6]]) {dst_order = #NCHW, mem_perm = #NCHW}
    // CHECK-SAME:       : tensor<1x64x512x64xf32, {order = #NHWC}> -> tensor<1x512x64x64xf32>

    // CHECK:        return [[PERMUTECAST_2]] : tensor<1x512x64x64xf32>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// MVN1NormalizeOp without from_force_decompose (default false): came from the normal decompose
// path in DecomposeMVNPass (internalReshape or CMX-overflow MVN). These were already
// Shave-efficient; converting them to DPU creates serial tile overhead with no Shave savings.
// Verify it is NOT converted to NCEMaxPool.
//
// CHECK-LABEL: func.func @NotReplaceMVN1NormalizeNormalDecomposePath
func.func @NotReplaceMVN1NormalizeNormalDecomposePath(%arg0: tensor<1x512x64x64xf32>) -> tensor<1x512x64x64xf32> {
    %0 = VPU.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x512x64x64xf32> -> tensor<1x64x512x64xf32, {order = #NHWC}>
    %1 = VPU.ShapeCast {shape = [1, 512, 64, 64]} inputs(%0 : tensor<1x64x512x64xf32, {order = #NHWC}>) -> tensor<1x512x64x64xf32, {order = #NHWC}>
    %2 = VPU.Convert(%1) {dstElemType = f16} : tensor<1x512x64x64xf32, {order = #NHWC}> -> tensor<1x512x64x64xf16, {order = #NHWC}>
    %3 = VPU.ShapeCast {shape = [1, 512, 4096, 1]} inputs(%2 : tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x512x4096x1xf16, {order = #NHWC}>
    %4 = VPU.MVN1SumOp(%3) {across_channels = false, normalize_variance = true, output_height = 3 : i64} : tensor<1x512x4096x1xf16, {order = #NHWC}> -> tensor<1x512x3x2xf32, {order = #NHWC}>
    %5 = VPU.MVN1MeanVar(%4) {across_channels = false, eps = 9.9999999999999995E-7 : f64, internal_reshape = [1, 32, 16, 4096], normalize_variance = true, orig_shape = [1, 512, 64, 64], output_type = f16} : tensor<1x512x3x2xf32, {order = #NHWC}> -> tensor<1x512x1x2xf16, {order = #NHWC}>
    %6 = VPU.MVN1Normalize(%3, %5) {across_channels = false, normalize_variance = true} : tensor<1x512x4096x1xf16, {order = #NHWC}>, tensor<1x512x1x2xf16, {order = #NHWC}> -> tensor<1x512x4096x1xf16, {order = #NHWC}>
    %7 = VPU.ShapeCast {shape = [1, 512, 64, 64]} inputs(%6 : tensor<1x512x4096x1xf16, {order = #NHWC}>) -> tensor<1x512x64x64xf16, {order = #NHWC}>
    %8 = VPU.Convert(%7) {dstElemType = f32} : tensor<1x512x64x64xf16, {order = #NHWC}> -> tensor<1x512x64x64xf32, {order = #NHWC}>
    %9 = VPU.ShapeCast {shape = [1, 64, 512, 64]} inputs(%8 : tensor<1x512x64x64xf32, {order = #NHWC}>) -> tensor<1x64x512x64xf32, {order = #NHWC}>
    %10 = VPU.PermuteCast(%9) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x512x64xf32, {order = #NHWC}> -> tensor<1x512x64x64xf32>
    return %10 : tensor<1x512x64x64xf32>

    // from_force_decompose is absent (defaults to false): normal decompose path.
    // Pass must leave MVN1Normalize as SHAVE -- no NCEMaxPool replacement.
    // CHECK:     VPU.MVN1Normalize
    // CHECK-NOT: VPU.NCE.MaxPool
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// MVN1NormalizeOp with from_force_decompose = true but f32 input: NCE.Eltwise and
// NCE.MaxPool only accept f16/bf16, so the element-type guard must block conversion
// and leave the op as SHAVE.
// Verify it is NOT converted to NCEMaxPool.
//
// CHECK-LABEL: func.func @NotReplaceMVN1NormalizeF32Input
func.func @NotReplaceMVN1NormalizeF32Input(
        %arg0: tensor<1x512x4096x1xf32, {order = #NHWC}>,
        %arg1: tensor<1x512x1x2xf16, {order = #NHWC}>) -> tensor<1x512x4096x1xf32, {order = #NHWC}> {
    %0 = VPU.MVN1Normalize(%arg0, %arg1) {across_channels = false, normalize_variance = true, from_force_decompose = true}
         : tensor<1x512x4096x1xf32, {order = #NHWC}>, tensor<1x512x1x2xf16, {order = #NHWC}>
         -> tensor<1x512x4096x1xf32, {order = #NHWC}>
    return %0 : tensor<1x512x4096x1xf32, {order = #NHWC}>

    // f32 input is not supported by NCE -- must remain SHAVE.
    // CHECK:     VPU.MVN1Normalize
    // CHECK-NOT: VPU.NCE.MaxPool
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// MVN1NormalizeOp from forceDecompose path where the ORIGINAL MVN tensor was 1D (W=1):
// DecomposeMVN flattens NHWC tensors to [1,C,H*W,1]; for W=1, the input is unchanged.
// The orig_shape attribute in MVN1MeanVar records the pre-flatten shape; W=1 there means
// the tensor was a 1D temporal signal. Converting to NCE.MaxPool serializes the DPU
// pipeline (blocks the next DPU convolution) with no spatial throughput benefit.
// The op must remain on SHAVE.
//
// CHECK-LABEL: func.func @NotReplaceMVN1Normalize1DSpatial
func.func @NotReplaceMVN1Normalize1DSpatial(
        %arg0: tensor<1x16x1551x1xf16, {order = #NHWC}>) -> tensor<1x16x1551x1xf16, {order = #NHWC}> {
    %sum = VPU.MVN1SumOp(%arg0) {across_channels = false, normalize_variance = true, output_height = 3 : i64}
           : tensor<1x16x1551x1xf16, {order = #NHWC}> -> tensor<1x16x3x2xf32, {order = #NHWC}>
    %meanvar = VPU.MVN1MeanVar(%sum) {across_channels = false, eps = 9.9999999999999995E-7 : f64,
                                      normalize_variance = true,
                                      orig_shape = [1, 16, 1551, 1], output_type = f16}
               : tensor<1x16x3x2xf32, {order = #NHWC}> -> tensor<1x16x1x2xf16, {order = #NHWC}>
    %norm = VPU.MVN1Normalize(%arg0, %meanvar) {across_channels = false, normalize_variance = true,
                                                from_force_decompose = true}
            : tensor<1x16x1551x1xf16, {order = #NHWC}>, tensor<1x16x1x2xf16, {order = #NHWC}>
            -> tensor<1x16x1551x1xf16, {order = #NHWC}>
    return %norm : tensor<1x16x1551x1xf16, {order = #NHWC}>

    // orig_shape W=1: originally 1D temporal -- DPU NCE.MaxPool serializes pipeline, must stay as SHAVE.
    // CHECK:     VPU.MVN1Normalize
    // CHECK-NOT: VPU.NCE.MaxPool
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// AddBroadcast with dynamic (non-constant) bias, NHWC layout, 2D spatial (H=16 W=16,
// area=256 >= threshold), f16 inputs, and channel-aligned C=16: should be converted
// to NCE.MaxPool with per-channel weights table (scale=1, bias=conditioning_bias).
//
// CHECK-LABEL: func.func @ReplaceAddBroadcastWithMaxPool
// CHECK-SAME:    [[FEATURE:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>
// CHECK-SAME:    [[BIAS:%.+]]: tensor<1x16x1x1xf16, {order = #NHWC}>
func.func @ReplaceAddBroadcastWithMaxPool(
        %arg0: tensor<1x16x16x16xf16, {order = #NHWC}>,
        %arg1: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = VPU.Add(%arg0, %arg1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // The pass generates ops in this order: scale-const, then the bias fp32 conversion via a DPU
    // identity NCE.MaxPool (convertToFp32ViaIdentityMaxPool) -- inner identity weight table
    // (scale=1, bias=0), inner NCE.MaxPool, reshape back to [16,1,1,1] -- then the outer weight
    // table (scale=CST_SCALE, bias=inner-maxpool-result) and the main NCE.MaxPool.
    // CHECK:        [[CST_SCALE:%.+]] = const.Declare tensor<16x1x1x1xf32, {order = #NHWC}> = dense<1.000000e+00>

    // CHECK-DAG:    [[BIAS_ONE:%.+]] = const.Declare tensor<16x1x1x1xf32, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-DAG:    [[BIAS_ZERO:%.+]] = const.Declare tensor<16x1x1x1xf32, {order = #NHWC}> = dense<0.000000e+00>
    // CHECK-DAG:    [[BIAS_WT_ZEROS:%.+]] = const.Declare tensor<16x1x1x2xf32, {order = #NHWC}> = dense<0.000000e+00>

    // CHECK:        [[BIAS_WT_CONCAT:%.+]] = VPU.Concat([[BIAS_WT_ZEROS]], [[BIAS_ONE]], [[BIAS_ZERO]])

    // CHECK:        [[BIAS_WT:%.+]] = Core.ReinterpretCast([[BIAS_WT_CONCAT]])
    // CHECK-SAME:       -> tensor<16x1x1x4xsi32, {order = #NHWC}>

    // CHECK:        [[BIAS_MAXPOOL:%.+]] = VPU.NCE.MaxPool([[BIAS]], [[BIAS_WT]]
    // CHECK-SAME:       -> tensor<1x16x1x1xf32, {order = #NHWC}>

    // CHECK:        [[BIAS_SHAPE_CAST:%.+]] = VPU.ShapeCast {shape = [16, 1, 1, 1]} inputs([[BIAS_MAXPOOL]] : tensor<1x16x1x1xf32, {order = #NHWC}>)
    // CHECK-SAME:       -> tensor<16x1x1x1xf32, {order = #NHWC}>

    // CHECK:        [[CST_ZEROS:%.+]] = const.Declare tensor<16x1x1x2xf32, {order = #NHWC}> = dense<0.000000e+00>

    // CHECK:        [[CONCAT:%.+]] = VPU.Concat([[CST_ZEROS]], [[CST_SCALE]], [[BIAS_SHAPE_CAST]])
    // CHECK-SAME:       {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 3]]}

    // CHECK:        [[REINTERPRET:%.+]] = Core.ReinterpretCast([[CONCAT]])
    // CHECK-SAME:       : tensor<16x1x1x4xf32, {order = #NHWC}> -> tensor<16x1x1x4xsi32, {order = #NHWC}>

    // CHECK:        [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[FEATURE]], [[REINTERPRET]] )
    // CHECK-SAME:       -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:        return [[MAXPOOL]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// AddBroadcast where the bias comes from a compile-time constant (const.Declare).
// isConstantBias() detects this and the pass must leave the Add unchanged.
//
// CHECK-LABEL: func.func @NotReplaceAddBroadcastConstBias
func.func @NotReplaceAddBroadcastConstBias(%arg0: tensor<1x16x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<1x16x1x1xf16>, [#const.Reorder<#NHWC>]
    %0 = VPU.Add(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
         : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // Constant bias: isConstantBias() returns true -- pass must leave Add unchanged.
    // CHECK: VPU.Add
    // CHECK-NOT: VPU.NCE.MaxPool
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// AddBroadcast where the feature map has H=1 (1D-like spatial data): NCEMaxPool
// DPU dispatch overhead for 1D sequential data outweighs the broadcast benefit.
// The H==1 guard must reject this and leave the Add unchanged.
//
// CHECK-LABEL: func.func @NotReplaceAddBroadcastOneDSpatial
func.func @NotReplaceAddBroadcastOneDSpatial(
        %arg0: tensor<1x16x1x256xf16, {order = #NHWC}>,
        %arg1: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x1x256xf16, {order = #NHWC}> {
    %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
         : tensor<1x16x1x256xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x1x256xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x256xf16, {order = #NHWC}>

    // H=1 feature map: not 2D spatial -- must remain as Add.
    // CHECK: VPU.Add
    // CHECK-NOT: VPU.NCE.MaxPool
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// AddBroadcast with f32 feature map: NCE.MaxPool only accepts f16/bf16 inputs.
// The element-type guard must reject this and leave the Add unchanged.
//
// CHECK-LABEL: func.func @NotReplaceAddBroadcastF32
func.func @NotReplaceAddBroadcastF32(
        %arg0: tensor<1x16x16x16xf32, {order = #NHWC}>,
        %arg1: tensor<1x16x1x1xf32, {order = #NHWC}>) -> tensor<1x16x16x16xf32, {order = #NHWC}> {
    %0 = VPU.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
         : tensor<1x16x16x16xf32, {order = #NHWC}>, tensor<1x16x1x1xf32, {order = #NHWC}> -> tensor<1x16x16x16xf32, {order = #NHWC}>
    return %0 : tensor<1x16x16x16xf32, {order = #NHWC}>

    // f32 input: NCE.MaxPool only accepts f16/bf16 -- must remain as Add.
    // CHECK: VPU.Add
    // CHECK-NOT: VPU.NCE.MaxPool
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// AddBroadcast where the sole consumer is VPU.LeakyRelu: the look-ahead in RunAddBroadcastOnDPU
// fuses the activation into the NCEMaxPool PPE as mode=LPRELU, replacing both VPU.Add and the
// downstream VPU.LeakyRelu with a single NCE.MaxPool op.
// This is the VPU-level half of the K2 AdaIN LeakyRelu fix: the IE-level pass strips the post_op
// from the broadcast Add and inserts a standalone IE.LeakyRelu; when lowered to VPU, this pattern
// becomes a clean VPU.Add (no post_op) followed by VPU.LeakyRelu, which the look-ahead fuses here.
//
// CHECK-LABEL: func.func @ReplaceAddBroadcastWithMaxPoolFusedLeakyRelu
// CHECK-SAME:    [[FEATURE:%.+]]: tensor<1x16x16x16xf16, {order = #NHWC}>
// CHECK-SAME:    [[BIAS:%.+]]: tensor<1x16x1x1xf16, {order = #NHWC}>
func.func @ReplaceAddBroadcastWithMaxPoolFusedLeakyRelu(
        %arg0: tensor<1x16x16x16xf16, {order = #NHWC}>,
        %arg1: tensor<1x16x1x1xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %0 = VPU.Add(%arg0, %arg1) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x16x16xf16, {order = #NHWC}>, tensor<1x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    %1 = VPU.LeakyRelu(%0) {negative_slope = 1.000000e-02 : f64}
         : tensor<1x16x16x16xf16, {order = #NHWC}> -> tensor<1x16x16x16xf16, {order = #NHWC}>
    return %1 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // VPU.Add's sole consumer is VPU.LeakyRelu: the look-ahead fuses the activation into the
    // NCEMaxPool PPE (mode=LPRELU) and erases the standalone VPU.LeakyRelu.
    // The return value becomes the NCEMaxPool output directly.
    // CHECK-NOT:    VPU.Add
    // CHECK:        [[MAXPOOL:%.+]] = VPU.NCE.MaxPool([[FEATURE]], {{%.+}})
    // CHECK-SAME:       ppe = #VPU.PPEFp<mode = <LPRELU>,
    // CHECK-SAME:       -> tensor<1x16x16x16xf16, {order = #NHWC}>
    // CHECK-NOT:    VPU.LeakyRelu
    // CHECK:        return [[MAXPOOL]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}
