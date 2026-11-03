//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --scf-vertical-fusion="vf-merge-configuration=GREEDY" --resolve-shaped-type-result-dims --canonicalize --create-workloads-for-nce-ops-scf %s | FileCheck %s
// REQUIRES: platform-NPU4000

// End-to-end check that DPU workload generation is correct for SCF loops that are
// produced by Vertical Fusion, rather than hand-written. VF emits an scf.for whose
// body carries a chain of NCE ops with bounded-dynamic tiled outputs, where the tile
// width of each op is back-inferred from its consumer. Every op in the chain therefore
// sees a *different* set of possible runtime widths, and the workload pass must
// enumerate exactly those widths per op instead of assuming a single shape.

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @VFChainWorkloads
func.func @VFChainWorkloads(%arg0: tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x128x540x240xf16, {order = #NHWC}>
 {
    %cst = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_2 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_3 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_4 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_5 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %cst_9 = const.Declare tensor<32x16x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x32x1x1xf32>, [#const.Reshape<[32, 1, 1, 1]>, #const.CastElemType<f16>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<#NHWC>]
    %0 = VPU.ShapeCast {shape = [1, 32, 540, 960]} inputs(%arg0 : tensor<1x256x540x120xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %cst) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %cst_0) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_1) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %4 = VPU.NCE.DepthConvolution(%3, %cst_9) rawFilterShape [32, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %5 = VPU.NCE.Convolution(%4, %cst_2) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %6 = VPU.NCE.Convolution(%5, %cst_3) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %7 = VPU.NCE.Convolution(%6, %cst_4) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 21]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %8 = VPU.NCE.DepthConvolution(%7, %cst_5) rawFilterShape [32, 1, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 20]} -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %9 = VPU.ShapeCast {shape = [1, 128, 540, 240]} inputs(%8 : tensor<1x32x540x960xf16, {order = #NHWC}>) -> tensor<1x128x540x240xf16, {order = #NHWC}>

    return %9: tensor<1x128x540x240xf16, {order = #NHWC}>

    // VF fuses the whole chain into a single scf.for tiled over W with step 26.
    // CHECK:       [[C0:%.+]] = arith.constant 0 : index
    // CHECK:       scf.for {{%.+}} = [[C0]] to {{%.+}} step {{%.+}} iter_args

    // The head of the chain consumes the widest back-inferred slice, so it has the
    // largest set of runtime widths. Each candidate is selected by an arith.cmpi on
    // the dynamic dim and dispatched through a chain of scf.if / else.
    // CHECK:       [[DIM0:%.+]] = tensor.dim
    // CHECK:       [[C38:%.+]] = arith.constant 38 : index
    // CHECK:       [[EQ38:%.+]] = arith.cmpi eq, [[DIM0]], [[C38]] : index
    // CHECK:       [[C37:%.+]] = arith.constant 37 : index
    // CHECK:       [[EQ37:%.+]] = arith.cmpi eq, [[DIM0]], [[C37]] : index
    // CHECK:       [[C33:%.+]] = arith.constant 33 : index
    // CHECK:       [[EQ33:%.+]] = arith.cmpi eq, [[DIM0]], [[C33]] : index

    // CHECK:       VPU.NCE.Convolution
    // The bound is the widest per-tile width (38), not the full 960 of the untiled tensor:
    // tile bounds are tightened per VF chain element, so the chain narrows 38 -> 36 -> ... -> 28.
    // CHECK-SAME:      -> tensor<1x32x540x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 540, 36]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:         scf.if [[EQ38]] {
    // CHECK:           VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 32, 541, 38] outOffsets [0, 0, 0, 0] outSizes [1, 32, 540, 36] pad [0, 0, 0, 0] <CUBOID_16x16>
    // CHECK:         } else {
    // CHECK:           scf.if [[EQ37]] {
    // CHECK:             VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 32, 541, 37] outOffsets [0, 0, 0, 0] outSizes [1, 32, 540, 35] pad [0, 0, 0, 0] <CUBOID_16x16>
    // CHECK:           } else {
    // CHECK:             scf.if [[EQ33]] {
    // CHECK:               VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 32, 541, 33] outOffsets [0, 0, 0, 0] outSizes [1, 32, 540, 31] pad [0, 0, 0, 0] <CUBOID_16x16>
    // CHECK:             } else {
    // CHECK:               VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 32, 541, 32] outOffsets [0, 0, 0, 0] outSizes [1, 32, 540, 30] pad [0, 0, 0, 0] <CUBOID_16x16>

    // Moving down the chain each conv sheds the 2 columns of halo consumed by its
    // 3x3 kernel, so the candidate set shrinks. The last conv is down to two widths.
    // CHECK:       [[C28:%.+]] = arith.constant 28 : index
    // CHECK:       [[EQ28:%.+]] = arith.cmpi eq, {{%.+}}, [[C28]] : index
    // CHECK:       VPU.NCE.Convolution
    // CHECK:         scf.if [[EQ28]] {
    // CHECK:           VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 32, 541, 28] outOffsets [0, 0, 0, 0] outSizes [1, 32, 540, 26] pad [0, 0, 0, 0] <CUBOID_16x16>
    // CHECK:         } else {
    // CHECK:           VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 32, 541, 27] outOffsets [0, 0, 0, 0] outSizes [1, 32, 540, 25] pad [0, 0, 0, 0] <CUBOID_16x16>

    // The 1x1 depthwise tail needs no halo, so in and out widths match and only the
    // loop's own two tile widths (full 26 / remainder 25) remain.
    // CHECK:       [[C26:%.+]] = arith.constant 26 : index
    // CHECK:       [[EQ26:%.+]] = arith.cmpi eq, {{%.+}}, [[C26]] : index
    // CHECK:       VPU.NCE.DepthConvolution
    // CHECK:         scf.if [[EQ26]] {
    // CHECK:           VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 32, 540, 26] outOffsets [0, 0, 0, 0] outSizes [1, 32, 540, 26] pad [0, 0, 0, 0] <CUBOID_16x16>
    // CHECK:         } else {
    // CHECK:           VPU.DPU.Workload inOffsets [0, 0, 0, 0] inSizes [1, 32, 540, 25] outOffsets [0, 0, 0, 0] outSizes [1, 32, 540, 25] pad [0, 0, 0, 0] <CUBOID_16x16>

    // CHECK:       tensor.insert_slice
    // CHECK:       scf.yield
}

// -----

// VF chain with skip connections and multi-use values inside the loop body.
// Workload generation temporarily retypes NCE operands to per-tile shapes, so
// multi-use operands must remain valid and receive per-tile workloads.

config.Resources 3 of @NCE at 1.700000e+03 MHz {
    config.MemoryResource 1474560 bytes of @CMX_NN {config.bandwidth = 64 : i64, config.derateFactor = 1.000000e+00 : f64}
    config.ExecutorResource 2 of @SHAVE_ACT
    config.ExecutorResource 1 of @DPU
}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @VFSkipConnectionWorkloads(%arg0: tensor<1x32x540x960xf16, {order = #NHWC}>) -> tensor<1x32x540x960xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_0 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %cst_1 = const.Declare tensor<32x32x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]

    %1 = VPU.NCE.Convolution(%arg0, %cst) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %2 = VPU.NCE.Convolution(%1, %cst_0) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %cst_1) rawFilterShape [32, 32, 3, 3] {resultSegmentSizes = array<i32: 1, 0, 0, 0>, mpe_engine = #VPU.MPEEngine37XX<mode = <SCL>>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, pad = #VPU.Padding<left = 1 : i64, right = 1 : i64, top = 1 : i64, bottom = 1 : i64>, ppe = #VPU.PPEInt<mode = <LRELU>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, fp_prelu_alpha = 1.000000e+00 : f64>,  strides = [1, 1], tilingStrategy = [1, 1, 1, 22]} : tensor<1x32x540x960xf16, {order = #NHWC}>, tensor<32x32x3x3xf16, {order = #NHWC}> -> tensor<1x32x540x960xf16, {order = #NHWC}>

    // Skip connection: consumes the chain tail %3 together with the earlier %1.
    %4 = VPU.NCE.Eltwise(%3, %1) {resultSegmentSizes = array<i32: 1, 0, 0, 0>, multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEInt<mode = <NOOP>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>, tilingStrategy = [1, 1, 1, 22]} -> tensor<1x32x540x960xf16, {order = #NHWC}>

    return %4: tensor<1x32x540x960xf16, {order = #NHWC}>
}

// CHECK-LABEL: @VFSkipConnectionWorkloads

// Vertical Fusion must have produced a single loop carrying the whole chain.
// CHECK: scf.for

// The first convolution is multi-use: padded into the next conv, and re-sliced for the
// skip connection. It must still get per-tile workloads, never the full 960 width.
// CHECK:       [[CONV0:%.+]] = VPU.NCE.Convolution
// CHECK:         VPU.DPU.Workload
// CHECK-NOT:     outSizes [1, 32, 540, 960]

// The skip slice is taken from the first convolution's result, confirming the multi-use
// operand survived workload generation with its original type intact.
// CHECK:       tensor.extract_slice [[CONV0]]
// CHECK-SAME:    skip_connection_slice

// The eltwise consumes the chain tail together with the skip slice, and is itself given
// per-tile workloads.
// CHECK:       VPU.NCE.Eltwise
// CHECK:         VPU.DPU.Workload
// CHECK-NOT:     outSizes [1, 32, 540, 960]
