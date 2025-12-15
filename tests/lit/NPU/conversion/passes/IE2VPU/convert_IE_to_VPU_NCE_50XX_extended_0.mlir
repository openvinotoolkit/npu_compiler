//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW allow-custom-values=true enable-auto-padding-odu=true enable-is-reduce-supported=true" --mlir-elide-elementsattrs-if-larger 64 --convert-IE-to-VPU-NCE %s | FileCheck %s
// REQUIRES: arch-NPU50XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TestReduceMeanWithAttr
module @TestReduceMeanWithAttr {
    // CHECK-LABEL:    func.func @ReduceMeanWithAttrNCE
    // CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x30x30xf16, {order = #NHWC}>)
    func.func @ReduceMeanWithAttrNCE(%arg0: tensor<1x16x30x30xf16, {order = #NHWC}>) -> tensor<1x16x30x30xf16, {order = #NHWC}> {
        %0 = IE.ReduceMean(%arg0) {axes_value = [1], keep_dims, output_padding = [0, 15, 0, 0]} : tensor<1x16x30x30xf16, {order = #NHWC}> -> tensor<1x16x30x30xf16, {order = #NHWC}>
        return %0 : tensor<1x16x30x30xf16, {order = #NHWC}>

        // CHECK:       [[MEAN:%.+]] =  VPU.NCE.Reduce([[INPUT]]) {
        // CHECK-SAME:    axes = [1], op_type = #VPU.reduce_type<MEAN>, output_padding = [0, 15, 0, 0],
        // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 6.250000e-02 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
        // CHECK-SAME:  } -> tensor<1x16x30x30xf16, {order = #NHWC}>

        // CHECK-NEXT:   return [[MEAN]] : tensor<1x16x30x30xf16, {order = #NHWC}>
    }

    // CHECK-LABEL:    func.func @ReduceSumWithAttrNCE
    // CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x30x30xf16, {order = #NHWC}>)
    func.func @ReduceSumWithAttrNCE(%arg0: tensor<1x16x30x30xf16, {order = #NHWC}>) -> tensor<1x16x30x30xf16, {order = #NHWC}> {
        %0 = IE.ReduceSum(%arg0) {axes_value = [1], keep_dims, output_padding = [0, 15, 0, 0]} : tensor<1x16x30x30xf16, {order = #NHWC}> -> tensor<1x16x30x30xf16, {order = #NHWC}>
        return %0 : tensor<1x16x30x30xf16, {order = #NHWC}>

        // CHECK:       [[MEAN:%.+]] =  VPU.NCE.Reduce([[INPUT]]) {
        // CHECK-SAME:    axes = [1], op_type = #VPU.reduce_type<SUM>, output_padding = [0, 15, 0, 0],
        // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <NOOP>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
        // CHECK-SAME:  } -> tensor<1x16x30x30xf16, {order = #NHWC}>

        // CHECK-NEXT:   return [[MEAN]] : tensor<1x16x30x30xf16, {order = #NHWC}>
    }

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TestReduceMeanWithConst
module @TestReduceMeanWithConst{
    // CHECK-LABEL:    func.func @ReduceMeanWithConstNCE
    // CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x30x30xf16, {order = #NHWC}>)
    func.func @ReduceMeanWithConstNCE(%arg0: tensor<1x16x30x30xf16, {order = #NHWC}>) -> tensor<1x16x30x30xf16, {order = #NHWC}> {
        %axes = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
        %0 = IE.ReduceMean(%arg0, %axes) {keep_dims, output_padding = [0, 15, 0, 0]} : tensor<1x16x30x30xf16, {order = #NHWC}>, tensor<1xsi32> -> tensor<1x16x30x30xf16, {order = #NHWC}>
        return %0 : tensor<1x16x30x30xf16, {order = #NHWC}>

        // CHECK:       [[MEAN:%.+]] =  VPU.NCE.Reduce([[INPUT]]) {
        // CHECK-SAME:    axes = [1], op_type = #VPU.reduce_type<MEAN>, output_padding = [0, 15, 0, 0],
        // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <NOOP>,
        // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
        // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
        // CHECK-SAME:          scale = 6.250000e-02 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>

        // CHECK-SAME:  } -> tensor<1x16x30x30xf16, {order = #NHWC}>

        // CHECK-NEXT:  return [[MEAN]] : tensor<1x16x30x30xf16, {order = #NHWC}>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TestReduceMeanWithConstInputPadding
module @TestReduceMeanWithConstInputPadding{
    // CHECK-LABEL:    func.func @ReduceMeanWithConstNCEInputPadding
    // CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x32x40x40xf16, {order = #NHWC}>)
    func.func @ReduceMeanWithConstNCEInputPadding(%arg0: tensor<1x32x40x40xf16, {order = #NHWC}>) -> tensor<1x16x40x40xf16, {order = #NHWC}> {
        %axes = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi32>
        %0 = IE.ReduceMean(%arg0, %axes) {keep_dims, input_padding = [0, 4, 0, 0], output_padding = [0, 15, 0, 0]} : tensor<1x32x40x40xf16, {order = #NHWC}>, tensor<1xsi32> -> tensor<1x16x40x40xf16, {order = #NHWC}>
        return %0 : tensor<1x16x40x40xf16, {order = #NHWC}>

        // CHECK:       [[MEAN:%.+]] =  VPU.NCE.Reduce([[INPUT]]) {
        // CHECK-SAME:    axes = [1], input_padding = [0, 4, 0, 0], op_type = #VPU.reduce_type<MEAN>, output_padding = [0, 15, 0, 0],
        // CHECK-SAME:    ppe = #VPU.PPEFp<mode = <NOOP>,
        // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
        // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
        // CHECK-SAME:          scale = 0.035714285714285712 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>

        // CHECK-SAME:  } -> tensor<1x16x40x40xf16, {order = #NHWC}>

        // CHECK-NEXT:  return [[MEAN]] : tensor<1x16x40x40xf16, {order = #NHWC}>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseSubtractToNCE
// CHECK-SAME:     ([[INPUT_0:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @EltwiseSubtractToNCE(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = IE.Subtract(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<SUBTRACT>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK:       } -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.078431372549019607>
!qElemType1 = !quant.uniform<u8:f16, 0.039215686274509803>

// CHECK-LABEL: @EltwiseMultiplyWithDifferentScales
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x1x2xui8, {order = #NHWC}>)
func.func @EltwiseMultiplyWithDifferentScales(%arg0: tensor<1x16x1x2xui8, {order = #NHWC}>) -> tensor<1x16x1x2xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x16x1x2x!qElemType, {order = #NHWC}> =
        // TODO: #-126284 Revert this back to dense<1>
        dense<2.000000e+00> : tensor<1x16x1x2xf32>, [
            #const.CastElemType<f16>,
            #const.CastElemType<ui8>,
            #const.CastElemType<!qElemType>,
            #const.Reorder<#NHWC>
        ]

    %0 = IE.QuantizeCast(%arg0) {
        dstElemType = !qElemType1
    } : tensor<1x16x1x2xui8, {order = #NHWC}> -> tensor<1x16x1x2x!qElemType1, {order = #NHWC}>

    %1 = IE.Multiply(%0, %cst) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x1x2x!qElemType1, {order = #NHWC}>, tensor<1x16x1x2x!qElemType, {order = #NHWC}> -> tensor<1x16x1x2xf16, {order = #NHWC}>

    return %1 : tensor<1x16x1x2xf16, {order = #NHWC}>

    // CHECK-DAG:       [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x16x1x2x!qElemType, {order = #NHWC}> =
    // TODO: #-126284 Revert this back to dense<1>
    // CHECK-SAME:  dense<2.000000e+00> : tensor<1x16x1x2xf32>, [
    // CHECK-SAME:    #const.CastElemType<f16>,
    // CHECK-SAME:    #const.CastElemType<ui8>,
    // CHECK-SAME:    #const.CastElemType<!qElemType>,
    // CHECK-SAME:    #const.Reorder<#NHWC>
    // CHECK-SAME:  ]

    // CHECK:       [[QUANT_CAST:%.*]] = IE.QuantizeCast([[INPUT]]) {
    // CHECK-SAME:      dstElemType = !qElemType1
    // CHECK-SAME:  } : tensor<1x16x1x2xui8, {order = #NHWC}> -> tensor<1x16x1x2x!qElemType1, {order = #NHWC}>

    // CHECK:       [[NCE_ADD:%.*]] = VPU.NCE.Eltwise([[QUANT_CAST]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      op_type = #VPU.eltwise_type<MULTIPLY>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.9073486328125E-6 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64,
    // CHECK-SAME:          in1_mult = [2.056000e+04], in2_mult = [4.112000e+04]>
    // CHECK-SAME:  } -> tensor<1x16x1x2xf16, {order = #NHWC}>

    // CHECK:   return [[NCE_ADD]] : tensor<1x16x1x2xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, -0.01013327205882353>
!qElemType1 = !quant.uniform<u8:f16, 0.053278186274509802>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-DAG: [[QELEMTYPE_IN:!.+]] = !quant.uniform<u8:f16, -0.01013327205882353>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<u8:f16, 0.053278186274509802>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @EltwiseMultiplyWithNegativeScale
// CHECK-SAME:     ([[INPUT_0:%.+]]: tensor<1x64x28x28x[[QELEMTYPE_IN]], {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x28x28x[[QELEMTYPE_IN]], {order = #NHWC}>)
func.func @EltwiseMultiplyWithNegativeScale(%arg0: tensor<1x64x28x28x!qElemType, {order = #NHWC}>, %arg1: tensor<1x64x28x28x!qElemType, {order = #NHWC}>)
        -> tensor<1x64x28x28x!qElemType1, {order = #NHWC}> {
    %0 = IE.Multiply(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28x!qElemType, {order = #NHWC}>, tensor<1x64x28x28x!qElemType, {order = #NHWC}>
        -> tensor<1x64x28x28x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x64x28x28x!qElemType1, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<MULTIPLY>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 2.550000e+02 : f64,
    // CHECK-SAME:          scale = 8.9495442807674407E-6 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64,
    // CHECK-SAME:          in1_mult = [2.125200e+04], in2_mult = [2.125200e+04]>
    // CHECK-SAME:      } -> tensor<1x64x28x28x[[QELEMTYPE_OUT]], {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28x[[QELEMTYPE_OUT]], {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, -0.049356617647058822>
!qElemType1 = !quant.uniform<u8:f16, -0.01013327205882353>
!qElemType2 = !quant.uniform<u8:f16, 0.053278186274509802>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-DAG: [[QELEMTYPE_LHS:!.+]] = !quant.uniform<u8:f16, -0.049356617647058822>
// CHECK-DAG: [[QELEMTYPE_RHS:!.+]] = !quant.uniform<u8:f16, -0.01013327205882353>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<u8:f16, 0.053278186274509802>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @EltwiseMultiplyWithDifferentNegativeScales
// CHECK-SAME:     ([[INPUT_0:%.+]]: tensor<1x64x28x28x[[QELEMTYPE_LHS]], {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x28x28x[[QELEMTYPE_RHS]], {order = #NHWC}>)
func.func @EltwiseMultiplyWithDifferentNegativeScales(%arg0: tensor<1x64x28x28x!qElemType, {order = #NHWC}>, %arg1: tensor<1x64x28x28x!qElemType1, {order = #NHWC}>)
        -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}> {
    %0 = IE.Multiply(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28x!qElemType, {order = #NHWC}>, tensor<1x64x28x28x!qElemType1, {order = #NHWC}>
        -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    return %0 : tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<MULTIPLY>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 2.550000e+02 : f64,
    // CHECK-SAME:          scale = 3.5798177123069763E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64,
    // CHECK-SAME:          in1_mult = [2.587800e+04], in2_mult = [5.313000e+03]>
    // CHECK-SAME:      } -> tensor<1x64x28x28x[[QELEMTYPE_OUT]], {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28x[[QELEMTYPE_OUT]], {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, -0.049356617647058822>
!qElemType1 = !quant.uniform<u8:f16, 0.01013327205882353>
!qElemType2 = !quant.uniform<u8:f16, 0.053278186274509802>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-DAG: [[QELEMTYPE_LHS:!.+]] = !quant.uniform<u8:f16, -0.049356617647058822>
// CHECK-DAG: [[QELEMTYPE_RHS:!.+]] = !quant.uniform<u8:f16, 0.01013327205882353>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<u8:f16, 0.053278186274509802>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @EltwiseMultiplyWithMixedNegativeScales
// CHECK-SAME:     ([[INPUT_0:%.+]]: tensor<1x64x28x28x[[QELEMTYPE_LHS]], {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x28x28x[[QELEMTYPE_RHS]], {order = #NHWC}>)
func.func @EltwiseMultiplyWithMixedNegativeScales(%arg0: tensor<1x64x28x28x!qElemType, {order = #NHWC}>, %arg1: tensor<1x64x28x28x!qElemType1, {order = #NHWC}>)
        -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}> {
    %0 = IE.Multiply(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28x!qElemType, {order = #NHWC}>, tensor<1x64x28x28x!qElemType1, {order = #NHWC}>
        -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    return %0 : tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<MULTIPLY>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 2.550000e+02 : f64,
    // CHECK-SAME:          scale = -3.5798177123069763E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64,
    // CHECK-SAME:          in1_mult = [2.587800e+04], in2_mult = [5.312000e+03]>
    // CHECK-SAME:      } -> tensor<1x64x28x28x[[QELEMTYPE_OUT]], {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28x[[QELEMTYPE_OUT]], {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddWithReluRewriter
// CHECK-SAME:     ([[INPUT_0:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @EltwiseAddWithReluRewriter(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, post_op = #IE.Relu<> } :
        tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LRELU>
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [-0.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>

    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x1x4xf16, {order = #NHWC}>)
func.func @MaxPoolToNCE(%arg0: tensor<1x16x1x4xf16, {order = #NHWC}>) -> tensor<1x16x1x4xf16, {order = #NHWC}> {
    %0 = IE.MaxPool(%arg0) {
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>,
            post_op = #IE.Clamp<min = 0.000000e+00 : f64, max = 6.000000e+00 : f64>
        } : tensor<1x16x1x4xf16, {order = #NHWC}> -> tensor<1x16x1x4xf16, {order = #NHWC}>

    return %0 : tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.MaxPool([[INPUT]]) {kernel_size = [1, 1],
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LRELUX>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 6.000000e+00 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      -> tensor<1x16x1x4xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x16x1x4xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MaxPoolFP32ToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x1x4xf16, {order = #NHWC}>)
func.func @MaxPoolFP32ToNCE(%arg0: tensor<1x16x1x4xf16, {order = #NHWC}>) -> tensor<1x16x1x4xf32, {order = #NHWC}> {
    %0 = IE.MaxPool(%arg0) {
            kernel_size = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            rounding_type = #IE.rounding_type<FLOOR>,
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
        } : tensor<1x16x1x4xf16, {order = #NHWC}> -> tensor<1x16x1x4xf32, {order = #NHWC}>

    return %0 : tensor<1x16x1x4xf32, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.MaxPool([[INPUT]]) {kernel_size = [1, 1],
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      -> tensor<1x16x1x4xf32, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x16x1x4xf32, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AveragePoolToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @AveragePoolToNCE(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x14x14xf16, {order = #NHWC}> {
    %0 = IE.AvgPool(%arg0) {
            kernel_size = [2, 2],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            rounding_type = #IE.rounding_type<FLOOR>,
            strides = [2, 2]
         } : tensor<1x64x28x28xf16, {order = #NHWC}> -> tensor<1x64x14x14xf16, {order = #NHWC}>

    return %0 : tensor<1x64x14x14xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {kernel_size = [2, 2],
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64
    // CHECK-SAME:          scale = 2.500000e-01 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      strides = [2, 2]}
    // CHECK-SAME:      -> tensor<1x64x14x14xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x14x14xf16, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.049356617647058822>
!qElemType1 = !quant.uniform<u8:f16, 0.01013327205882353>
!qElemType2 = !quant.uniform<u8:f16, 0.053278186274509802>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddWithDifferentScales
// CHECK-SAME:     ([[INPUT_0:%.+]]: tensor<1x64x28x28x!qElemType, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x28x28x!qElemType1, {order = #NHWC}>)
func.func @EltwiseAddWithDifferentScales(%arg0: tensor<1x64x28x28x!qElemType, {order = #NHWC}>, %arg1: tensor<1x64x28x28x!qElemType1, {order = #NHWC}>)
        -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28x!qElemType, {order = #NHWC}>, tensor<1x64x28x28x!qElemType1, {order = #NHWC}>
        -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    return %0 : tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,

    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64
    // CHECK-SAME:          clamp_high = 2.550000e+02 : f64
    // CHECK-SAME:          scale = 3.5798177123069763E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64
    // CHECK-SAME:          in1_mult = [2.587700e+04], in2_mult = [5.312000e+03]>
    // CHECK-SAME:      } -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28x!qElemType2, {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, -0.01013327205882353>
!qElemType1 = !quant.uniform<u8:f16, 0.053278186274509802>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-DAG: [[QELEMTYPE_IN:!.+]] = !quant.uniform<u8:f16, -0.01013327205882353>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<u8:f16, 0.053278186274509802>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @EltwiseAddWithNegativeScale
// CHECK-SAME:     ([[INPUT_0:%.+]]: tensor<1x64x28x28x[[QELEMTYPE_IN]], {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x28x28x[[QELEMTYPE_IN]], {order = #NHWC}>)
func.func @EltwiseAddWithNegativeScale(%arg0: tensor<1x64x28x28x!qElemType, {order = #NHWC}>, %arg1: tensor<1x64x28x28x!qElemType, {order = #NHWC}>)
        -> tensor<1x64x28x28x!qElemType1, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28x!qElemType, {order = #NHWC}>, tensor<1x64x28x28x!qElemType, {order = #NHWC}>
        -> tensor<1x64x28x28x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x64x28x28x!qElemType1, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 2.550000e+02 : f64,
    // CHECK-SAME:          scale = -8.9495442807674407E-6 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64,
    // CHECK-SAME:          in1_mult = [2.125200e+04], in2_mult = [2.125200e+04]>
    // CHECK-SAME:      } -> tensor<1x64x28x28x[[QELEMTYPE_OUT]], {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28x[[QELEMTYPE_OUT]], {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, -0.049356617647058822>
!qElemType1 = !quant.uniform<u8:f16, -0.01013327205882353>
!qElemType2 = !quant.uniform<u8:f16, 0.053278186274509802>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-DAG: [[QELEMTYPE_LHS:!.+]] = !quant.uniform<u8:f16, -0.049356617647058822>
// CHECK-DAG: [[QELEMTYPE_RHS:!.+]] = !quant.uniform<u8:f16, -0.01013327205882353>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<u8:f16, 0.053278186274509802>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @EltwiseAddWithDifferentNegativeScales
// CHECK-SAME:     ([[INPUT_0:%.+]]: tensor<1x64x28x28x[[QELEMTYPE_LHS]], {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x28x28x[[QELEMTYPE_RHS]], {order = #NHWC}>)
func.func @EltwiseAddWithDifferentNegativeScales(%arg0: tensor<1x64x28x28x!qElemType, {order = #NHWC}>, %arg1: tensor<1x64x28x28x!qElemType1, {order = #NHWC}>)
        -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28x!qElemType, {order = #NHWC}>, tensor<1x64x28x28x!qElemType1, {order = #NHWC}>
        -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    return %0 : tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 2.550000e+02 : f64,
    // CHECK-SAME:          scale = -3.5798177123069763E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64,
    // CHECK-SAME:          in1_mult = [2.587800e+04], in2_mult = [5.313000e+03]>
    // CHECK-SAME:      } -> tensor<1x64x28x28x[[QELEMTYPE_OUT]], {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28x[[QELEMTYPE_OUT]], {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, -0.049356617647058822>
!qElemType1 = !quant.uniform<u8:f16, 0.01013327205882353>
!qElemType2 = !quant.uniform<u8:f16, 0.053278186274509802>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-DAG: [[QELEMTYPE_LHS:!.+]] = !quant.uniform<u8:f16, -0.049356617647058822>
// CHECK-DAG: [[QELEMTYPE_RHS:!.+]] = !quant.uniform<u8:f16, 0.01013327205882353>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<u8:f16, 0.053278186274509802>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @KeepAddWithMixedNegativeScales
// CHECK-SAME:     ([[INPUT_0:%.+]]: tensor<1x64x28x28x[[QELEMTYPE_LHS]], {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x28x28x[[QELEMTYPE_RHS]], {order = #NHWC}>)
func.func @KeepAddWithMixedNegativeScales(%arg0: tensor<1x64x28x28x!qElemType, {order = #NHWC}>, %arg1: tensor<1x64x28x28x!qElemType1, {order = #NHWC}>)
        -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28x!qElemType, {order = #NHWC}>, tensor<1x64x28x28x!qElemType1, {order = #NHWC}>
        -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    return %0 : tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = IE.Add([[INPUT_0]], [[INPUT_1]])
    // CHECK-SAME:    tensor<1x64x28x28x[[QELEMTYPE_LHS]], {order = #NHWC}>, tensor<1x64x28x28x[[QELEMTYPE_RHS]], {order = #NHWC}>
    // CHECK-SAME:    -> tensor<1x64x28x28x[[QELEMTYPE_OUT]], {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28x[[QELEMTYPE_OUT]], {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, -0.049356617647058822>
!qElemType1 = !quant.uniform<u8:f16, -0.01013327205882353>
!qElemType2 = !quant.uniform<u8:f16, 0.053278186274509802>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-DAG: [[QELEMTYPE_LHS:!.+]] = !quant.uniform<u8:f16, -0.049356617647058822>
// CHECK-DAG: [[QELEMTYPE_RHS:!.+]] = !quant.uniform<u8:f16, -0.01013327205882353>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<u8:f16, 0.053278186274509802>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @EltwiseSubtractWithDifferentNegativeScales
// CHECK-SAME:     ([[INPUT_0:%.+]]: tensor<1x64x28x28x[[QELEMTYPE_LHS]], {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x28x28x[[QELEMTYPE_RHS]], {order = #NHWC}>)
func.func @EltwiseSubtractWithDifferentNegativeScales(%arg0: tensor<1x64x28x28x!qElemType, {order = #NHWC}>, %arg1: tensor<1x64x28x28x!qElemType1, {order = #NHWC}>)
        -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}> {
    %0 = IE.Subtract(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28x!qElemType, {order = #NHWC}>, tensor<1x64x28x28x!qElemType1, {order = #NHWC}>
        -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    return %0 : tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<SUBTRACT>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 2.550000e+02 : f64,
    // CHECK-SAME:          scale = -3.5798177123069763E-5 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64,
    // CHECK-SAME:          in1_mult = [2.587800e+04], in2_mult = [5.313000e+03]>
    // CHECK-SAME:      } -> tensor<1x64x28x28x[[QELEMTYPE_OUT]], {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28x[[QELEMTYPE_OUT]], {order = #NHWC}>
}

// -----

!qElemType = !quant.uniform<u8:f16, -0.049356617647058822>
!qElemType1 = !quant.uniform<u8:f16, 0.01013327205882353>
!qElemType2 = !quant.uniform<u8:f16, 0.053278186274509802>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-DAG: [[QELEMTYPE_LHS:!.+]] = !quant.uniform<u8:f16, -0.049356617647058822>
// CHECK-DAG: [[QELEMTYPE_RHS:!.+]] = !quant.uniform<u8:f16, 0.01013327205882353>
// CHECK-DAG: [[QELEMTYPE_OUT:!.+]] = !quant.uniform<u8:f16, 0.053278186274509802>

// Note that "CHECK-LABEL" directive is deliberately skipped here because it resets previously captured variables
// CHECK:       @KeepSubtractWithMixedNegativeScales
// CHECK-SAME:     ([[INPUT_0:%.+]]: tensor<1x64x28x28x[[QELEMTYPE_LHS]], {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x28x28x[[QELEMTYPE_RHS]], {order = #NHWC}>)
func.func @KeepSubtractWithMixedNegativeScales(%arg0: tensor<1x64x28x28x!qElemType, {order = #NHWC}>, %arg1: tensor<1x64x28x28x!qElemType1, {order = #NHWC}>)
        -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}> {
    %0 = IE.Subtract(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28x!qElemType, {order = #NHWC}>, tensor<1x64x28x28x!qElemType1, {order = #NHWC}>
        -> tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    return %0 : tensor<1x64x28x28x!qElemType2, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = IE.Subtract([[INPUT_0]], [[INPUT_1]])
    // CHECK-SAME:    tensor<1x64x28x28x[[QELEMTYPE_LHS]], {order = #NHWC}>, tensor<1x64x28x28x[[QELEMTYPE_RHS]], {order = #NHWC}>
    // CHECK-SAME:    -> tensor<1x64x28x28x[[QELEMTYPE_OUT]], {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28x[[QELEMTYPE_OUT]], {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SkipEltwiseAndToNCE
// CHECK-SAME:     ([[INPUT_0:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @SkipEltwiseAndToNCE(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = IE.And(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK-NOT:   VPU.NCE.Eltwise

    // CHECK:       [[OUT:%.+]] = IE.And([[INPUT_0]], [[INPUT_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} :
    // CHECK-SAME:      tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCE4channels
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x4x16x16xf16, {order = #NHWC}>)
func.func @ConvToNCE4channels(%arg0: tensor<1x4x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x4x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x4x1x1xf16>, [#const.Reorder<#NHWC>]
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
        } : tensor<1x4x16x16xf16, {order = #NHWC}>, tensor<16x4x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x1x1x16xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x4x1x1xf16>,
    // CHECK-SAME:      [#const.Reorder<#NHWC>, #const.Reshape<[16, 1, 1, 4]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 12]>]
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.CompressConvolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      cm_sp_pattern = 15 : i64,
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias = 1.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      rawFilterShape = [16, 4, 1, 1]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.078431372549019607>
!qElemType1 = !quant.uniform<u8:f16, 0.039215686274509803>

// CHECK-LABEL: @AddWithDifferentScales
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x1x2xui8, {order = #NHWC}>)
func.func @AddWithDifferentScales(%arg0: tensor<1x16x1x2xui8, {order = #NHWC}>) -> tensor<1x16x1x2xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<1x16x1x2x!qElemType, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<1x16x1x2xf32>, [
            #const.CastElemType<f16>,
            #const.CastElemType<ui8>,
            #const.CastElemType<!qElemType>,
            #const.Reorder<#NHWC>
        ]

    %0 = IE.QuantizeCast(%arg0) {
        dstElemType = !qElemType1
    } : tensor<1x16x1x2xui8, {order = #NHWC}> -> tensor<1x16x1x2x!qElemType1, {order = #NHWC}>

    %1 = IE.Add(%0, %cst) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x16x1x2x!qElemType1, {order = #NHWC}>, tensor<1x16x1x2x!qElemType, {order = #NHWC}> -> tensor<1x16x1x2xf16, {order = #NHWC}>

    return %1 : tensor<1x16x1x2xf16, {order = #NHWC}>

    // CHECK-DAG:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x16x1x2x!qElemType, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<1x16x1x2xf32>, [
    // CHECK-SAME:    #const.CastElemType<f16>,
    // CHECK-SAME:    #const.CastElemType<ui8>,
    // CHECK-SAME:    #const.CastElemType<!qElemType>,
    // CHECK-SAME:    #const.Reorder<#NHWC>
    // CHECK-SAME:  ]

    // CHECK:   [[QUANT_CAST:%.*]] = IE.QuantizeCast([[INPUT]]) {
    // CHECK-SAME:      dstElemType = !qElemType1
    // CHECK-SAME:  } : tensor<1x16x1x2xui8, {order = #NHWC}> -> tensor<1x16x1x2x!qElemType1, {order = #NHWC}>

    // CHECK:   [[NCE_ADD:%.*]] = VPU.NCE.Eltwise([[QUANT_CAST]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:     op_type = #VPU.eltwise_type<ADD>,
    // CHECK-SAME:     ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.9073486328125E-6 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64,
    // CHECK-SAME:          in1_mult = [2.056000e+04], in2_mult = [4.112000e+04]>
    // CHECK-SAME:     } -> tensor<1x16x1x2xf16, {order = #NHWC}>

    // CHECK:   return [[NCE_ADD]] : tensor<1x16x1x2xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @ConvertPermuteQuantize
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x224x224xf16>)
func.func @ConvertPermuteQuantize(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 1, 0, 0]
    } : tensor<1x3x224x224xf16> -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   IE.PermuteQuantize

    // CHECK:       [[NCE_PERMUTE:%.*]] = VPU.NCE.Permute([[INPUT]]) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dstOrder = #NHWC,
    // CHECK-SAME:      expandedChannels = 4 : i64
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 2.550000e+02 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>

    // CHECK-SAME:  } -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK:       return [[NCE_PERMUTE]] : tensor<1x4x224x224x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 0.0078431372549019607:0>

// CHECK-LABEL: @ConvertPermuteQuantizeWithQuantU8Input
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x224x224x!qElemType>)
func.func @ConvertPermuteQuantizeWithQuantU8Input(%arg0: tensor<1x3x224x224x!qElemType>) -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 1, 0, 0]
    } : tensor<1x3x224x224x!qElemType> -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   IE.PermuteQuantize

    // CHECK:       [[NCE_PERMUTE:%.+]] = VPU.NCE.Permute([[INPUT]]) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dstOrder = #NHWC,
    // CHECK-SAME:      expandedChannels = 4 : i64
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 2.550000e+02 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>

    // CHECK-SAME:  } -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK:       return [[NCE_PERMUTE]] : tensor<1x4x224x224x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @PermuteQuantizeDoesNotFitCMX
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x512x512xf16>)
func.func @PermuteQuantizeDoesNotFitCMX(%arg0: tensor<1x3x512x512xf16>) -> tensor<1x16x512x512x!qElemType, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 13, 0, 0]
    } : tensor<1x3x512x512xf16> -> tensor<1x16x512x512x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x16x512x512x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   IE.PermuteQuantize

    // CHECK:       [[NCE_PERMUTE:%.*]] = VPU.NCE.Permute([[INPUT]]) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dstOrder = #NHWC,
    // CHECK-SAME:      expandedChannels = 16 : i64
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = 0.000000e+00 : f64,
    // CHECK-SAME:          clamp_high = 2.550000e+02 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>

    // CHECK-SAME:  } -> tensor<1x16x512x512x!qElemType, {order = #NHWC}>

    // CHECK:       return [[NCE_PERMUTE]] : tensor<1x16x512x512x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @PermuteQuantizeStartPadsOverHeight
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x224x224xf16>)
func.func @PermuteQuantizeStartPadsOverHeight(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x225x224x!qElemType, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 1, 0],
        pads_end = [0, 1, 0, 0]
    } : tensor<1x3x224x224xf16> -> tensor<1x4x225x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x225x224x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   VPU.NCE.Permute
    // CHECK-NOT:   VPU.Reshape
    // CHECK-NOT:   IE.AffineReshape

    // CHECK:       [[PERMUTE_QUANTIZE:%.*]] = IE.PermuteQuantize([[INPUT]]) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NHWC,
    // CHECK-SAME:      pads_begin = [0, 0, 1, 0],
    // CHECK-SAME:      pads_end = [0, 1, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x224x224xf16> -> tensor<1x4x225x224x!qElemType, {order = #NHWC}>

    // CHECK:       return [[PERMUTE_QUANTIZE]] : tensor<1x4x225x224x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @PermuteQuantizeEndPadsOverHeight
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x224x224xf16>)
func.func @PermuteQuantizeEndPadsOverHeight(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x225x224x!qElemType, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 1, 1, 0]
    } : tensor<1x3x224x224xf16> -> tensor<1x4x225x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x225x224x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   VPU.NCE.Permute
    // CHECK-NOT:   VPU.Reshape
    // CHECK-NOT:   IE.AffineReshape

    // CHECK:       [[PERMUTE_QUANTIZE:%.*]] = IE.PermuteQuantize([[INPUT]]) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NHWC,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 1, 1, 0]
    // CHECK-SAME:  } : tensor<1x3x224x224xf16> -> tensor<1x4x225x224x!qElemType, {order = #NHWC}>

    // CHECK:       return [[PERMUTE_QUANTIZE]] : tensor<1x4x225x224x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @PermuteQuantizeUnsupportedInputLayout
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x224x224xf16, {order = #NCWH}>)
func.func @PermuteQuantizeUnsupportedInputLayout(%arg0: tensor<1x3x224x224xf16, {order = #NCWH}>) -> tensor<1x4x224x224x!qElemType, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 1, 0, 0]
    } : tensor<1x3x224x224xf16, {order = #NCWH}> -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   VPU.NCE.Permute
    // CHECK-NOT:   VPU.Reshape
    // CHECK-NOT:   IE.AffineReshape

    // CHECK:       [[PERMUTE_QUANTIZE:%.*]] = IE.PermuteQuantize([[INPUT]]) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NHWC,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 1, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x224x224xf16, {order = #NCWH}> -> tensor<1x4x224x224x!qElemType, {order = #NHWC}>

    // CHECK:       return [[PERMUTE_QUANTIZE]] : tensor<1x4x224x224x!qElemType, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @PermuteQuantizeUnsupportedOutputLayout
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x224x224xf16>)
func.func @PermuteQuantizeUnsupportedOutputLayout(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x4x224x224x!qElemType, {order = #NWCH}> {
    %0 = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NWCH,
        mem_perm = #NWCH,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 1, 0, 0]
    } : tensor<1x3x224x224xf16> -> tensor<1x4x224x224x!qElemType, {order = #NWCH}>

    return %0 : tensor<1x4x224x224x!qElemType, {order = #NWCH}>

    // CHECK-NOT:   VPU.NCE.Permute
    // CHECK-NOT:   VPU.Reshape
    // CHECK-NOT:   IE.AffineReshape

    // CHECK:       [[PERMUTE_QUANTIZE:%.*]] = IE.PermuteQuantize([[INPUT]]) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dst_order = #NWCH,
    // CHECK-SAME:      mem_perm = #NWCH,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 1, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x224x224xf16> -> tensor<1x4x224x224x!qElemType, {order = #NWCH}>

    // CHECK:       return [[PERMUTE_QUANTIZE]] : tensor<1x4x224x224x!qElemType, {order = #NWCH}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<u8:f16, 1.000000e+00>

// CHECK-LABEL: @PermuteQuantizeUnsupportedShape
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x3x225x225xf16>)
func.func @PermuteQuantizeUnsupportedShape(%arg0: tensor<1x3x225x225xf16>) -> tensor<1x4x225x225x!qElemType, {order = #NHWC}> {
    %0 = IE.PermuteQuantize(%arg0) {
        dstElemType = !qElemType,
        dst_order = #NHWC,
        mem_perm = #NHWC,
        pads_begin = [0, 0, 0, 0],
        pads_end = [0, 1, 0, 0]
    } : tensor<1x3x225x225xf16> -> tensor<1x4x225x225x!qElemType, {order = #NHWC}>

    return %0 : tensor<1x4x225x225x!qElemType, {order = #NHWC}>

    // CHECK-NOT:   VPU.NCE.Permute
    // CHECK-NOT:   VPU.Reshape
    // CHECK-NOT:   IE.AffineReshape

    // CHECK:       [[PERMUTE_QUANTIZE:%.*]] = IE.PermuteQuantize([[INPUT]]) {
    // CHECK-SAME:      dstElemType = !qElemType,
    // CHECK-SAME:      dst_order = #NHWC,
    // CHECK-SAME:      mem_perm = #NHWC,
    // CHECK-SAME:      pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:      pads_end = [0, 1, 0, 0]
    // CHECK-SAME:  } : tensor<1x3x225x225xf16> -> tensor<1x4x225x225x!qElemType, {order = #NHWC}>

    // CHECK:       return [[PERMUTE_QUANTIZE]] : tensor<1x4x225x225x!qElemType, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCECompressConv
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x4x16x16xf16, {order = #NHWC}>)
func.func @ConvToNCECompressConv(%arg0: tensor<1x4x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x4x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>]
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
        } : tensor<1x4x16x16xf16, {order = #NHWC}>, tensor<16x4x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x1x1x16xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x3x1x1xf16>,
    // CHECK-SAME:          [#const.Reorder<#NHWC>, #const.Reshape<[16, 1, 1, 3]>,
    // CHECK-SAME:          #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 13]>]
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.CompressConvolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      cm_sp_pattern = 7 : i64,
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias = 1.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    // CHECK-SAME:      rawFilterShape = [16, 3, 1, 1]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ConvToNCECompressConvWithPadBefore
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x4x16x16xf16, {order = #NHWC}>)
func.func @ConvToNCECompressConvWithPadBefore(%arg0: tensor<1x4x16x16xf16, {order = #NHWC}>) -> tensor<1x16x16x16xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x4x1x1xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 1, 0, 0], [0, 0, 0, 0]>]
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>

    %0 = IE.Convolution(%arg0, %weights, %bias) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
        } : tensor<1x4x16x16xf16, {order = #NHWC}>, tensor<16x4x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16>
            -> tensor<1x16x16x16xf16, {order = #NHWC}>

    return %0 : tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x1x1x16xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<16x3x1x1xf16>,
    // CHECK-SAME:          [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 1, 0, 0], [0, 0, 0, 0]>,
    // CHECK-SAME:           #const.Reshape<[16, 1, 1, 4]>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 12]>]
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>

    // CHECK:       [[VAL0:%.+]] = VPU.NCE.CompressConvolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      cm_sp_pattern = 15 : i64,
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias = 1.000000e+00 : f64, adder = 0.000000e+00 : f64>,
    // CHECK-SAME:      rawFilterShape = [16, 4, 1, 1]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x16x16xf16, {order = #NHWC}>

    // CHECK:       return [[VAL0]] : tensor<1x16x16x16xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @AvgPoolToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x6x6xf16, {order = #NHWC}>)
func.func @AvgPoolToNCE(%arg0: tensor<1x16x6x6xf16, {order = #NHWC}>) -> tensor<1x16x4x4xf16, {order = #NHWC}> {
    %ave_pool = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [3, 3],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x6x6xf16, {order = #NHWC}> -> tensor<1x16x4x4xf16, {order = #NHWC}>

    return %ave_pool : tensor<1x16x4x4xf16, {order = #NHWC}>

    // CHECK:         [[OUT:%.+]] = VPU.NCE.AveragePool([[INPUT]]) {kernel_size = [3, 3]
    // CHECK-SAME:        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:        ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:           clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:           clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:           scale = 0.1111111111111111 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>

    // CHECK:           return [[OUT]] : tensor<1x16x4x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NotConvertAvgPoolToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x6x6xf16, {order = #NHWC}>)
func.func @NotConvertAvgPoolToNCE(%arg0: tensor<1x16x6x6xf16, {order = #NHWC}>) -> tensor<1x16x5x4xf16, {order = #NHWC}> {
    %ave_pool = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [3, 3],
        pads_begin = [1, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x16x6x6xf16, {order = #NHWC}> -> tensor<1x16x5x4xf16, {order = #NHWC}>

    return %ave_pool : tensor<1x16x5x4xf16, {order = #NHWC}>

    // CHECK:         [[OUT:%.+]] = IE.AvgPool([[INPUT]]) {exclude_pads, kernel_size = [3, 3], pads_begin = [1, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x6x6xf16, {order = #NHWC}> -> tensor<1x16x5x4xf16, {order = #NHWC}>
    // CHECK:           return [[OUT]] : tensor<1x16x5x4xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddToNCE
// CHECK-SAME:     ([[INPUT_0:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @EltwiseAddToNCE(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>, %arg1: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Eltwise([[INPUT_0]], [[INPUT_1]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @EltwiseAddSameInputsToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x64x28x28xf16, {order = #NHWC}>)
func.func @EltwiseAddSameInputsToNCE(%arg0: tensor<1x64x28x28xf16, {order = #NHWC}>)
        -> tensor<1x64x28x28xf16, {order = #NHWC}> {
    %0 = IE.Add(%arg0, %arg0) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<1x64x28x28xf16, {order = #NHWC}>, tensor<1x64x28x28xf16, {order = #NHWC}>
        -> tensor<1x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.Eltwise([[INPUT]], [[INPUT]])
    // CHECK-SAME:      op_type = #VPU.eltwise_type<ADD>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <NOOP>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e+00], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      -> tensor<1x64x28x28xf16, {order = #NHWC}>

    // CHECK:       return [[OUT]] : tensor<1x64x28x28xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvToNCE
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x40x80xf16, {order = #NHWC}>)
func.func @DepthConvToNCE(%arg0: tensor<1x16x40x80xf16, {order = #NHWC}>) -> tensor<1x16x37x73xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x1x4x8xf16>, [#const.Reorder<#NHWC>]

    %0 = IE.GroupConvolution(%arg0, %weights) {
            dilations = [1, 1],
            groups = 16,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
        } : tensor<1x16x40x80xf16, {order = #NHWC}>, tensor<16x1x4x8xf16, {order = #NHWC}>
            -> tensor<1x16x37x73xf16, {order = #NHWC}>

    return %0 : tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.DepthConvolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:                pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:                ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:                rawFilterShape = [16, 1, 4, 8], strides = [1, 1]}
    // CHECK-SAME:                 -> tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK-NEXT:      return [[OUT]] : tensor<1x16x37x73xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvToNCEWithWeightsAlign
// CHECK-SAME:     ([[INPUT_0:%.+]]: tensor<1x16x40x80xf16, {order = #NHWC}>,
// CHECK-SAME:      [[INPUT_1:%.+]]: tensor<16x1x2x3xf16, {order = #NHWC}>)
func.func @DepthConvToNCEWithWeightsAlign(%arg0: tensor<1x16x40x80xf16, {order = #NHWC}>, %arg1: tensor<16x1x2x3xf16, {order = #NHWC}>) -> tensor<1x16x39x78xf16, {order = #NHWC}> {
    %0 = IE.GroupConvolution(%arg0, %arg1) {
            dilations = [1, 1],
            groups = 16,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
        } : tensor<1x16x40x80xf16, {order = #NHWC}>, tensor<16x1x2x3xf16, {order = #NHWC}>
            -> tensor<1x16x39x78xf16, {order = #NHWC}>

    return %0 : tensor<1x16x39x78xf16, {order = #NHWC}>


    // CHECK-DAG:   [[CONST0:%.+]] = const.Declare tensor<16x10x1x1xf16> = dense<0.000000e+00> : tensor<16x10x1x1xf16>
    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>

    // CHECK-DAG:   [[SHAPECAST:%.+]] = IE.ShapeCast {shape = [16, 6, 1, 1]} inputs([[INPUT_1]] : tensor<16x1x2x3xf16, {order = #NHWC}>) -> tensor<16x6x1x1xf16, {order = #NHWC}>
    // CHECK-DAG:   [[PERMUTECAST:%.+]] = IE.PermuteCast([[SHAPECAST]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<16x6x1x1xf16, {order = #NHWC}> -> tensor<16x6x1x1xf16>
    // CHECK-DAG:   [[CONCAT:%.+]] = IE.Concat([[PERMUTECAST]], [[CONST0]]) {per_axis = #IE.Concat<axis = 1 : i64>} : tensor<16x6x1x1xf16>, tensor<16x10x1x1xf16> -> tensor<16x16x1x1xf16>
    // CHECK-DAG:   [[PERMUTECAST1:%.+]] = IE.PermuteCast([[CONCAT]]) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<16x16x1x1xf16> -> tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[DEPTHCONV:%.+]] = VPU.NCE.DepthConvolution([[INPUT_0]], [[PERMUTECAST1]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <LPRELU>, clamp_low = -3.4028234663852886E+38 : f64, clamp_high = 3.4028234663852886E+38 : f64, scale = 1.000000e+00 : f64, prelu_alpha = [1.000000e-01], bias = 0.000000e+00 : f64, adder = 0.000000e+00 : f64>
    // CHECK-SAME:      rawFilterShape = [16, 1, 2, 3], strides = [1, 1]}
    // CHECK-SAME:      -> tensor<1x16x39x78xf16, {order = #NHWC}>

    // CHECK:       return [[DEPTHCONV:%.+]] : tensor<1x16x39x78xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvToNCEAutopad
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x40x80xf16, {order = #NHWC}>)
func.func @DepthConvToNCEAutopad(%arg0: tensor<1x16x40x80xf16, {order = #NHWC}>) -> tensor<1x16x37x73xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x1x4x8xf16>, [#const.Reorder<#NHWC>]

    %0 = IE.GroupConvolution(%arg0, %weights) {
            dilations = [1, 1],
            groups = 3,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>,
            input_padding = [0, 13, 0, 0],
            output_padding = [0, 13, 0, 0]
        } : tensor<1x16x40x80xf16, {order = #NHWC}>, tensor<16x1x4x8xf16, {order = #NHWC}>
            -> tensor<1x16x37x73xf16, {order = #NHWC}>

    return %0 : tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}>

    // CHECK:       [[OUT:%.+]] = VPU.NCE.DepthConvolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          input_padding = [0, 13, 0, 0],
    // CHECK-SAME:          output_padding = [0, 13, 0, 0],
    // CHECK-SAME:          ppe = #VPU.PPEFp<mode = <LPRELU>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e-01],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64>
    // CHECK-SAME:          -> tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK-NEXT:      return [[OUT]] : tensor<1x16x37x73xf16, {order = #NHWC}>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DepthConvWithSprLUTTanh
// CHECK-SAME:     ([[INPUT:%.+]]: tensor<1x16x40x80xf16, {order = #NHWC}>)
func.func @DepthConvWithSprLUTTanh(%input: tensor<1x16x40x80xf16, {order = #NHWC}>) -> tensor<1x16x37x73xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x1x4x8xf16>, [#const.Reorder<#NHWC>]

    %output = IE.GroupConvolution(%input, %weights) {
            dilations = [1, 1],
            groups = 16,
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1],
            post_op = #IE.Tanh<>
        } : tensor<1x16x40x80xf16, {order = #NHWC}>, tensor<16x1x4x8xf16, {order = #NHWC}>
            -> tensor<1x16x37x73xf16, {order = #NHWC}>

    return %output : tensor<1x16x37x73xf16, {order = #NHWC}>

    // CHECK-DAG:   [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x1x4x8xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.DepthConvolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:      ppe = #VPU.PPEFp<mode = <TANH>,
    // CHECK-SAME:          clamp_low = -3.4028234663852886E+38 : f64,
    // CHECK-SAME:          clamp_high = 3.4028234663852886E+38 : f64,
    // CHECK-SAME:          scale = 1.000000e+00 : f64,
    // CHECK-SAME:          prelu_alpha = [1.000000e+00],
    // CHECK-SAME:          bias = 0.000000e+00 : f64,
    // CHECK-SAME:          adder = 0.000000e+00 : f64,
    // CHECK-SAME:          sprlut = dense_resource<__elided__>

    // CHECK-NEXT:      return [[OUTPUT]] : tensor<1x16x37x73xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DontConvertGroupConvToNCEIfDilatedConv
func.func @DontConvertGroupConvToNCEIfDilatedConv(%arg0: tensor<1x16x48x48xf16, {order = #NHWC}>) -> tensor<1x16x48x48xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<16x1x3x3xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x1x3x3xf16>, [#const.Reorder<#NHWC>]

    %0 = IE.GroupConvolution(%arg0, %weights) {
            dilations = [2, 2],
            groups = 16,
            pads_begin = [2, 2],
            pads_end = [2, 2],
            strides = [1, 1],
            post_op = #IE.LeakyRelu<negative_slope = 1.000000e-01 : f64>
        } : tensor<1x16x48x48xf16, {order = #NHWC}>, tensor<16x1x3x3xf16, {order = #NHWC}>
            -> tensor<1x16x48x48xf16, {order = #NHWC}>

    return %0 : tensor<1x16x48x48xf16, {order = #NHWC}>

    // CHECK-NOT: VPU.NCE.DepthConvolution
    // CHECK:     IE.GroupConvolution
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DontConvertMultiplyToNCEIfMultiBatch
func.func @DontConvertMultiplyToNCEIfMultiBatch(%arg0: tensor<2x64x28x28xf16, {order = #NHWC}>, %arg1: tensor<2x64x28x28xf16, {order = #NHWC}>) -> tensor<2x64x28x28xf16, {order = #NHWC}> {
    %0 = IE.Multiply(%arg0, %arg1) { auto_broadcast = #IE.auto_broadcast_type<NUMPY> } :
        tensor<2x64x28x28xf16, {order = #NHWC}>, tensor<2x64x28x28xf16, {order = #NHWC}>
        -> tensor<2x64x28x28xf16, {order = #NHWC}>

    return %0 : tensor<2x64x28x28xf16, {order = #NHWC}>

    // CHECK-NOT: VPU.NCE.Eltwise
    // CHECK: IE.Multiply
}
