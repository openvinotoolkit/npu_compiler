//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --move-view-ops-to-vf="workload-management-mode=FWLM_V1_PAGES" %s | FileCheck %s
// REQUIRES: platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType  = !quant.uniform<u8:f16, 0.0094078685723099058:128>
!qElemType1 = !quant.uniform<u8:f16, 0.0047039342861549529:128>

// CHECK-LABEL: @MoveChainQuantizeCastAffineReshape
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x128x256x1x!qElemType, {order = #NHWC}>
func.func @MoveChainQuantizeCastAffineReshape(%arg0: tensor<1x128x256x1x!qElemType, {order = #NHWC}>)
        -> tensor<1x128x64x4xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x128x256x1x!qElemType, {order = #NHWC}>,
                             %arg0 as %arg2: tensor<1x128x256x1x!qElemType, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x128x256x1x!qElemType, {order = #NHWC}> {
      %inner = VPU.NCE.Eltwise(%arg1, %arg2)
          {resultSegmentSizes = array<i32: 1, 0, 0, 0>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
          -> tensor<1x128x256x1x!qElemType, {order = #NHWC}>
      VPU.Yield %inner
    }
    // QuantizeCast: scalar quant parameter change only, shape unchanged.
    // hasOneUse = true (only input to AffineReshape).
    %1 = VPU.QuantizeCast(%0) {dstElemType = !qElemType1}
        : tensor<1x128x256x1x!qElemType, {order = #NHWC}> -> tensor<1x128x256x1x!qElemType1, {order = #NHWC}>
    // AffineReshape: H=256 splits to H=64, W=4 (ratio 4). C=128 is 1:1 (tiling on C OK).
    // hasOneUse = true (only input to ConsumerVF as a single operand).
    %2 = VPU.AffineReshape(%1) {dim_mapping = [[0], [1], [2, 3], [3]], shape_value = [1, 128, 64, 4]}
        : tensor<1x128x256x1x!qElemType1, {order = #NHWC}> -> tensor<1x128x64x4x!qElemType1, {order = #NHWC}>
    // ConsumerVF uses %2 as a single operand; the inner Eltwise reuses the block arg.
    %3 = VPU.VerticalFusion (%2 as %arg1: tensor<1x128x64x4x!qElemType1, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 2, 1, 1]} -> tensor<1x128x64x4xf16, {order = #NHWC}> {
      %inner = VPU.NCE.Eltwise(%arg1, %arg1)
          {resultSegmentSizes = array<i32: 1, 0, 0, 0>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
          -> tensor<1x128x64x4xf16, {order = #NHWC}>
      VPU.Yield %inner
    }
    return %3 : tensor<1x128x64x4xf16, {order = #NHWC}>

    // Consumer VF's block-arg type changes to ProducerVF's output type (!qElemType).
    // Both QuantizeCast and AffineReshape are cloned inside the new VF body.
    //CHECK:  [[VF0:%.+]] = VPU.VerticalFusion ([[INPUT]] as
    //CHECK-SAME: tilingStrategy = [1, 2, 1, 1]
    //CHECK:  [[VF1:%.+]] = VPU.VerticalFusion ([[VF0]] as [[ARG:%[^:]+]]: tensor<1x128x256x1x!qElemType, {order = #NHWC}>)
    //CHECK-SAME: attributes {tilingStrategy = [1, 2, 1, 1]}
    //CHECK:  [[QC:%.+]]  = VPU.QuantizeCast([[ARG]]) {dstElemType = !qElemType1}
    //CHECK:  [[AR:%.+]]  = VPU.AffineReshape([[QC]])
    //CHECK:  [[ELT:%.+]] = VPU.NCE.Eltwise([[AR]], [[AR]])
    //CHECK:  VPU.Yield [[ELT]]
    //CHECK:  return [[VF1]] : tensor<1x128x64x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType  = !quant.uniform<u8:f16, 0.0094078685723099058:128>
!qElemType1 = !quant.uniform<u8:f16, 0.0047039342861549529:128>

// CHECK-LABEL: @MoveChainQuantizeCastShapeCast
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x64x60x1x!qElemType, {order = #NHWC}>
func.func @MoveChainQuantizeCastShapeCast(%arg0: tensor<1x64x60x1x!qElemType, {order = #NHWC}>)
        -> tensor<1x64x20x3xf16, {order = #NHWC}> {
    %0 = VPU.VerticalFusion (%arg0 as %arg1: tensor<1x64x60x1x!qElemType, {order = #NHWC}>,
                             %arg0 as %arg2: tensor<1x64x60x1x!qElemType, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x64x60x1x!qElemType, {order = #NHWC}> {
      %inner = VPU.NCE.Eltwise(%arg1, %arg2)
          {resultSegmentSizes = array<i32: 1, 0, 0, 0>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
          -> tensor<1x64x60x1x!qElemType, {order = #NHWC}>
      VPU.Yield %inner
    }
    // QuantizeCast: scalar change only, shape unchanged.
    // hasOneUse = true (only input to ShapeCast).
    %1 = VPU.QuantizeCast(%0) {dstElemType = !qElemType1}
        : tensor<1x64x60x1x!qElemType, {order = #NHWC}> -> tensor<1x64x60x1x!qElemType1, {order = #NHWC}>
    // ShapeCast: H*W 60*1 → 20*3, total size preserved (3840).
    // hasOneUse = true (only input to ConsumerVF as a single operand).
    %2 = VPU.ShapeCast {shape = [1, 64, 20, 3]}
        inputs(%1 : tensor<1x64x60x1x!qElemType1, {order = #NHWC}>)
        -> tensor<1x64x20x3x!qElemType1, {order = #NHWC}>
    // ConsumerVF uses %2 as a single operand; the inner Eltwise reuses the block arg.
    %3 = VPU.VerticalFusion (%2 as %arg1: tensor<1x64x20x3x!qElemType1, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x64x20x3xf16, {order = #NHWC}> {
      %inner = VPU.NCE.Eltwise(%arg1, %arg1)
          {resultSegmentSizes = array<i32: 1, 0, 0, 0>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>}
          -> tensor<1x64x20x3xf16, {order = #NHWC}>
      VPU.Yield %inner
    }
    return %3 : tensor<1x64x20x3xf16, {order = #NHWC}>

    // Consumer VF's block-arg type changes to ProducerVF's output type (!qElemType).
    // Both QuantizeCast and ShapeCast are cloned inside the new VF body.
    //CHECK:  [[VF0:%.+]] = VPU.VerticalFusion ([[INPUT]] as
    //CHECK-SAME: tilingStrategy = [1, 1, 2, 1]
    //CHECK:  [[VF1:%.+]] = VPU.VerticalFusion ([[VF0]] as [[ARG:%[^:]+]]: tensor<1x64x60x1x!qElemType, {order = #NHWC}>)
    //CHECK-SAME: attributes {tilingStrategy = [1, 1, 2, 1]}
    //CHECK:  [[QC:%.+]]  = VPU.QuantizeCast([[ARG]]) {dstElemType = !qElemType1}
    //CHECK:  [[SC:%.+]]  = VPU.ShapeCast {shape = [1, 64, 20, 3]} inputs([[QC]]
    //CHECK:  [[ELT:%.+]] = VPU.NCE.Eltwise([[SC]], [[SC]])
    //CHECK:  VPU.Yield [[ELT]]
    //CHECK:  return [[VF1]] : tensor<1x64x20x3xf16, {order = #NHWC}>
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType  = !quant.uniform<u8:f16, 0.0094078685723099058:128>
!qElemType1 = !quant.uniform<u8:f16, 0.0047039342861549529:128>

// CHECK-LABEL: @NotMoveQuantizeCastNoParentVFOp
// CHECK-SAME:    ([[INPUT:%.+]]: tensor<1x32x48x48x!qElemType, {order = #NHWC}>
func.func @NotMoveQuantizeCastNoParentVFOp(%arg0: tensor<1x32x48x48x!qElemType, {order = #NHWC}>) -> tensor<1x32x48x48xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<32x32x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<32x32x1x1xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
    %0 = VPU.NCE.Convolution(%arg0, %cst) rawFilterShape [32, 32, 1, 1] {resultSegmentSizes = array<i32: 1, 0, 0, 0>,
            pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
            ppe = #VPU.PPEStub<>,
             strides = [1, 1]}
        : tensor<1x32x48x48x!qElemType, {order = #NHWC}>, tensor<32x32x1x1xf16, {order = #NHWC}>
        -> tensor<1x32x48x48x!qElemType, {order = #NHWC}>
    %1 = VPU.QuantizeCast(%0) {dstElemType = !qElemType1}
        : tensor<1x32x48x48x!qElemType, {order = #NHWC}> -> tensor<1x32x48x48x!qElemType1, {order = #NHWC}>
    %2 = VPU.VerticalFusion (%1 as %arg1: tensor<1x32x48x48x!qElemType1, {order = #NHWC}>,
                             %1 as %arg2: tensor<1x32x48x48x!qElemType1, {order = #NHWC}>)
        attributes {tilingStrategy = [1, 1, 2, 1]} -> tensor<1x32x48x48xf16, {order = #NHWC}> {
      %3 = VPU.NCE.Eltwise(%arg1, %arg2)
        {resultSegmentSizes = array<i32: 1, 0, 0, 0>, op_type = #VPU.eltwise_type<ADD>, ppe = #VPU.PPEStub<>} -> tensor<1x32x48x48xf16, {order = #NHWC}>
      VPU.Yield %3
    }
    return %2 : tensor<1x32x48x48xf16, {order = #NHWC}>

    //CHECK:  [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]]
    //CHECK:  [[QC:%.+]] = VPU.QuantizeCast([[CONV]])
    //CHECK:  VPU.VerticalFusion ([[QC]] as {{%.+}}: tensor<1x32x48x48x!qElemType1, {order = #NHWC}>, [[QC]] as {{%.+}}: tensor<1x32x48x48x!qElemType1, {order = #NHWC}>)
    //CHECK-SAME: attributes {tilingStrategy = [1, 1, 2, 1]}
    //CHECK-NOT: VPU.QuantizeCast
    //CHECK:  VPU.NCE.Eltwise
}
