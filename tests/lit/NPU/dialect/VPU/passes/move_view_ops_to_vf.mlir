//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --move-view-ops-to-vf %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @MoveGroupSparseTensor(%arg0: tensor<1x32x24x30xf16, {order = #NHWC}>) -> tensor<1x16x48x60xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x32x2x2xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x32x2x2xf16, {order = #NHWC}>
    %0 = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 24, 30], seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>, seDepth = 1 : i64, seSize = [32]} -> tensor<1x1x49x61xi32, {order = #NHWC}>
    %cst_0 = const.Declare tensor<1x32x49x61xi1, {order = #NHWC}> = dense<1> : tensor<1x32x49x61xi8>, [#const.Reorder<#NHWC>, #const.CastElemType<i1>]
    %1 = VPU.GroupSparseTensor(%arg0, %cst_0, %0) {seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>} -> !VPU.SparseTensor<data=tensor<1x32x24x30xf16, {order = #NHWC}>, sparsity_map=tensor<1x32x49x61xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x49x61xi32, {order = #NHWC}>, #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>>
    %2 = VPU.VerticalFusion (%1 as %arg1: !VPU.SparseTensor<data=tensor<1x32x24x30xf16, {order = #NHWC}>, sparsity_map=tensor<1x32x49x61xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x49x61xi32, {order = #NHWC}>, #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>>, %cst as %arg2: tensor<16x32x2x2xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x16x48x60xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Convolution(%arg1, %arg2) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [16, 32, 2, 2], strides = [1, 1]} : !VPU.SparseTensor<data=tensor<1x32x24x30xf16, {order = #NHWC}>, sparsity_map=tensor<1x32x49x61xi1, {order = #NHWC}>, storage_element_table=tensor<1x1x49x61xi32, {order = #NHWC}>, #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>>, tensor<16x32x2x2xf16, {order = #NHWC}> -> tensor<1x16x48x60xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    return %2 : tensor<1x16x48x60xf16, {order = #NHWC}>

    //CHECK:  [[SET:%.+]] = VPU.StorageElementTable {dataElemType = f16, dataShape = [1, 32, 24, 30], seAttr = #VPU.SEUpsampling<factors = [1, 1], padding = [1, 1, 1, 1]>, seDepth = 1 : i64, seSize = [32]} -> tensor<1x1x49x61xi32, {order = #NHWC}>
    //CHECK:  VPU.VerticalFusion ({{[^ ]+}} as [[INPUT:%.+]]: tensor<1x32x24x30xf16, {order = #NHWC}>, {{[^ ]+}} as [[SPARSITY_MAP:%.+]]: tensor<1x32x49x61xi1, {order = #NHWC}>, [[SET]] as [[SE_TABLE:%.+]]: tensor<1x1x49x61xi32, {order = #NHWC}>, {{[^ ]+}} as [[WEIGHTS:%.+]]: tensor<16x32x2x2xf16, {order = #NHWC}>)
    //CHECK-SAME: attributes {tilingStrategy = [1, 1, 2, 1]}
    //CHECK:  [[GST:%.+]] = VPU.GroupSparseTensor([[INPUT]], [[SPARSITY_MAP]], [[SE_TABLE]])
    //CHECK:  [[CONV:%.+]] = VPU.NCE.Convolution([[GST]], [[WEIGHTS]])
    //CHECK:  VPU.Yield [[CONV]]

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0094078685723099058:128>
!qElemType1 = !quant.uniform<u8:f16, 0.0047039342861549529:128>

// CHECK-LABEL: @MoveQuantizeCast
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x48x48x48x!qElemType, {order = #NHWC}>
func.func @MoveQuantizeCast(%arg0: tensor<1x48x48x48x!qElemType, {order = #NHWC}>) -> tensor<1x32x48x48xf16, {order = #NHWC}> {
    %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 32, 48, 48] : tensor<1x48x48x48x!qElemType, {order = #NHWC}> to tensor<1x32x48x48x!qElemType, {order = #NHWC}>
    %1 = VPU.QuantizeCast(%0) {dstElemType = !qElemType1} : tensor<1x32x48x48x!qElemType, {order = #NHWC}> -> tensor<1x32x48x48x!qElemType1, {order = #NHWC}>
    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x32x48x48x!qElemType1, {order = #NHWC}>, %1 as %arg2: tensor<1x32x48x48x!qElemType1, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x48x48xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Eltwise(%arg1, %arg2)
        {op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x32x48x48xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    return %2 : tensor<1x32x48x48xf16, {order = #NHWC}>

    //CHECK:  [[SLICE:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 48, 48]
    //CHECK:  VPU.VerticalFusion ([[SLICE]] as %arg1: tensor<1x32x48x48x!qElemType, {order = #NHWC}>)
    //CHECK-SAME: attributes {tilingStrategy = [1, 1, 2, 1]}
    //CHECK:  [[QUANTIZE_CAST:%.+]] = VPU.QuantizeCast(%arg1)
    //CHECK:  [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[QUANTIZE_CAST]], [[QUANTIZE_CAST]])
    //CHECK:  VPU.Yield [[ELTWISE]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL: @NotMoveLayoutCast
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x120x120x40xf16>
func.func @NotMoveLayoutCast(%arg0: tensor<1x120x120x40xf16>) -> tensor<1x40x120x120xf16> {
    %0 = VPU.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x120x120x40xf16> -> tensor<1x40x120x120xf16, {order = #NHWC}>
    %1 = VPU.ShapeCast {shape = [1, 16, 120, 300]} inputs(%0 : tensor<1x40x120x120xf16, {order = #NHWC}>) -> tensor<1x16x120x300xf16, {order = #NHWC}>
    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x16x120x300xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 3]} -> tensor<1x16x120x300xf16, {order = #NWCH}> {
      %7 = VPU.NCE.MaxPool(%arg1) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1]} -> tensor<1x16x120x300xf16, {order = #NWCH}> 
      VPU.Yield %7 
    }
    %3 = VPU.LayoutCast(%2) {dst_order = #NHWC} : tensor<1x16x120x300xf16, {order = #NWCH}> -> tensor<1x16x120x300xf16, {order = #NHWC}>
    %4 = VPU.VerticalFusion (%3 as %arg1: tensor<1x16x120x300xf16, {order = #NHWC}>) attributes {tilingStrategy = [1, 1, 1, 3]} -> tensor<1x16x120x300xf16, {order = #NWCH}> {
      %7 = VPU.NCE.MaxPool(%arg1) {kernel_size = [1, 1], pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>, strides = [1, 1]} -> tensor<1x16x120x300xf16, {order = #NWCH}> 
      VPU.Yield %7 
    }
    %5 = VPU.PermuteCast(%4) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x16x120x300xf16, {order = #NWCH}> -> tensor<1x300x16x120xf16>
    %6 = VPU.ShapeCast {shape = [1, 40, 120, 120]} inputs(%5 : tensor<1x300x16x120xf16>) -> tensor<1x40x120x120xf16>
    return %6 : tensor<1x40x120x120xf16>

    //CHECK:  [[PERMUTE_CAST_0:%.+]] = VPU.PermuteCast([[INPUT]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x120x120x40xf16> -> tensor<1x40x120x120xf16, {order = #NHWC}>
    //CHECK:  [[SHAPE_CAST_0:%.+]] = VPU.ShapeCast {shape = [1, 16, 120, 300]} inputs([[PERMUTE_CAST_0]] : tensor<1x40x120x120xf16, {order = #NHWC}>) -> tensor<1x16x120x300xf16, {order = #NHWC}>
    //CHECK:  [[VERTICAL_FUSION_0:%.+]] = VPU.VerticalFusion ([[SHAPE_CAST_0]] as %arg1: tensor<1x16x120x300xf16, {order = #NHWC}>)
    //CHECK:  [[MAXPOOL_0:%.+]] = VPU.NCE.MaxPool(%arg1)
    //CHECK:  VPU.Yield [[MAXPOOL_0]]
    //CHECK:  [[LAYOUT_CAST:%.+]] = VPU.LayoutCast([[VERTICAL_FUSION_0]]) {dst_order = #NHWC} : tensor<1x16x120x300xf16, {order = #NWCH}> -> tensor<1x16x120x300xf16, {order = #NHWC}>
    //CHECK:  [[VERTICAL_FUSION_1:%.+]] = VPU.VerticalFusion ([[LAYOUT_CAST]] as %arg1: tensor<1x16x120x300xf16, {order = #NHWC}>)
    //CHECK:  [[MAXPOOL_1:%.+]] = VPU.NCE.MaxPool(%arg1)
    //CHECK:  VPU.Yield [[MAXPOOL_1]]
    //CHECK:  [[PERMUTE_CAST_1:%.+]] = VPU.PermuteCast([[VERTICAL_FUSION_1]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x16x120x300xf16, {order = #NWCH}> -> tensor<1x300x16x120xf16>
    //CHECK:  [[SHAPE_CAST_1:%.+]] = VPU.ShapeCast {shape = [1, 40, 120, 120]} inputs([[PERMUTE_CAST_1]] : tensor<1x300x16x120xf16>) -> tensor<1x40x120x120xf16>
}
