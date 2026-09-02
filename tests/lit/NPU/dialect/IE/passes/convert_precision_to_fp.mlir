//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-precision-to-fp --canonicalize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

//
// The 'convert-precision-to-fp' pass:
//
//   * Updates both Function bodies and Function prototypes.
//   * It shouldn't touch user types defined in `net.NetworkInfo`.
//   * It should update types for `Constant` operation.
//

// CHECK-LABEL: @FP32toFP16
module @FP32toFP16 {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "data" : tensor<1x1000xf32>
        DataInfo "data" : tensor<1x1000xf32>
    }
    outputsInfo : {
        // CHECK: DataInfo "prob" : tensor<1x1000xf32>
        DataInfo "prob" : tensor<1x1000xf32>
    }

// CHECK: func.func @main([[ARG0:[^:]+]]: tensor<1x1000xf16>) -> tensor<1x1000xf16>
func.func @main(%arg0: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
    %prob = IE.SoftMax(%arg0) {axisInd = 1} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    // CHECK:       [[OUT:%.+]] = IE.SoftMax([[ARG0]])
    // CHECK-SAME:      tensor<1x1000xf16> -> tensor<1x1000xf16>

    return %prob : tensor<1x1000xf32>
    // CHECK: return [[OUT]] : tensor<1x1000xf16>
}

}

// -----

// CHECK-LABEL: @ConstantLayer
module @ConstantLayer {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
    }
    outputsInfo : {
        // CHECK: DataInfo "output" : tensor<1x2x2x2xf32>
        DataInfo "output" : tensor<1x2x2x2xf32>
    }

// CHECK: func.func @main() -> tensor<1x2x2x2xf16>
func.func @main() -> tensor<1x2x2x2xf32> {
    %0 = const.Declare tensor<1x2x2x2xf32> = dense<1.0> : tensor<1x2x2x2xf32>
    return %0 : tensor<1x2x2x2xf32>

    // CHECK-DAG:       [[OUT:%.+]] = const.Declare tensor<1x2x2x2xf16> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<1x2x2x2xf32>, [#const.CastElemType<f16>]
    // CHECK:       return [[OUT]] : tensor<1x2x2x2xf16>
}

}

// -----

// CHECK-LABEL: @SplatConstantsWithOverflow
module @SplatConstantsWithOverflow {
net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
    }
    outputsInfo : {
        // CHECK: DataInfo "overflow1" : tensor<1x2x2x2xf32>
        DataInfo "overflow1" : tensor<1x2x2x2xf32>
        // CHECK: DataInfo "overflow2" : tensor<1x2x2x2xf32>
        DataInfo "overflow2" : tensor<1x2x2x2xf32>
    }

    // CHECK: func.func @main() -> (tensor<1x2x2x2xf16>, tensor<1x2x2x2xf16>)
    func.func @main() -> (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) {
        %negative = const.Declare tensor<1x2x2x2xf32> = dense<-100000.0> : tensor<1x2x2x2xf32>
        %positive = const.Declare tensor<1x2x2x2xf32> = dense<100000.0> : tensor<1x2x2x2xf32>
        return %negative, %positive : tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>

        // CHECK-DAG: [[NEG:%.+]] = const.Declare {{.+}} = dense<-6.550400e+04> : tensor<1x2x2x2xf16>, [#const.CastElemType<f16>]
        // CHECK-DAG: [[POS:%.+]] = const.Declare {{.+}} = dense<6.550400e+04> : tensor<1x2x2x2xf16>, [#const.CastElemType<f16>]
        // CHECK: return [[NEG]], [[POS]]
    }
}

// -----

// CHECK-LABEL: @ArrayConstantsWithOverflow
module @ArrayConstantsWithOverflow {
net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
    }
    outputsInfo : {
        // CHECK: DataInfo "overflow" : tensor<1x1x1x2xf32>
        DataInfo "overflow" : tensor<1x1x1x2xf32>
    }

    // CHECK: func.func @main() -> tensor<1x1x1x2xf16>
    func.func @main() -> tensor<1x1x1x2xf32> {
        %array = const.Declare tensor<1x1x1x2xf32> = dense<[[[[-100000.0, 100000.0]]]]> : tensor<1x1x1x2xf32>
        return %array : tensor<1x1x1x2xf32>

        // E#160872: float16 non-splats behave *differently* from splats...

        // CHECK: [[ARR:%.+]] = const.Declare
        // CHECK-SAME{LITERAL}: dense<[[[[-1.000000e+05, 1.000000e+05]]]]> : tensor<1x1x1x2xf32>, [#const.CastElemType<f16>]
        // CHECK: return [[ARR]]
    }
}

// -----

// CHECK-LABEL: @I8ToFp16
module @I8ToFp16 {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "in1" : tensor<1xf16>
        // CHECK: DataInfo "in2" : tensor<1xf16>
        DataInfo "in1" : tensor<1xf16>
        DataInfo "in2" : tensor<1xf16>
    }
    outputsInfo : {
        // CHECK: DataInfo "out" : tensor<1xf16>
        DataInfo "out" : tensor<1xf16>
    }

// CHECK: func.func @main([[ARG0:[^:]+]]: tensor<1xf16>, [[ARG1:[^:]+]]: tensor<1xf16>) -> tensor<1xf16>
func.func @main(%arg0: tensor<1xf16>, %arg1: tensor<1xf16>) -> tensor<1xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = i8} : tensor<1xf16> -> tensor<1xi8>
    %1 = IE.Convert(%arg1) {dstElemType = i8} : tensor<1xf16> -> tensor<1xi8>
    %2 = IE.And(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xi8>, tensor<1xi8> -> tensor<1xi8>
    %3 = IE.Convert(%2) {dstElemType = f16} : tensor<1xi8> -> tensor<1xf16>
    return %3 : tensor<1xf16>

    // CHECK:  [[AND:%.+]] = IE.And([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf16>, tensor<1xf16> -> tensor<1xf16>
    // CHECK:  return [[AND]] : tensor<1xf16>
}

}

// -----

// CHECK-LABEL: @OneHot
module @OneHot {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "Parameter_2994" : tensor<4xsi32>
        DataInfo "Parameter_2994" : tensor<4xsi32>
    }
    outputsInfo : {
        // CHECK: DataInfo "OneHot_2998" : tensor<3x4xf32>
        DataInfo "OneHot_2998" : tensor<3x4xf32>
    }

// CHECK: func.func @main([[ARG0:[^:]+]]: tensor<4xsi32>) -> tensor<3x4xf16>
func.func @main(%arg0: tensor<4xsi32>) -> tensor<3x4xf32> {
    %0 = IE.OneHot(%arg0) {axis_attr = 0 : i64, depth_attr = 3 : i64,  mode = #IE.one_hot_mode<IGNORE_NEGATIVE>, off_value_attr = 0.000000e+00 : f64, on_value_attr = 1.000000e+00 : f64, operandSegmentSizes = array<i32: 1, 0, 0, 0>, outputType = f32} : tensor<4xsi32> -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>

    // CHECK:       [[RESHAPE:%.+]] = IE.AffineReshape([[ARG0]])
    // CHECK-SAME:      shape_value = [1, 4]
    // CHECK-SAME:      tensor<4xsi32> -> tensor<1x4xsi32>
    // CHECK:       [[OUT:%.+]] = IE.OneHot([[RESHAPE]])
    // CHECK-SAME:      axis_attr = 0 : i64
    // CHECK-SAME:      depth_attr = 3 : i64
    // CHECK-SAME:      outputType = f16
    // CHECK-SAME:      rank_expanded = true
    // CHECK-SAME:      tensor<1x4xsi32> -> tensor<3x4xf16>
    // CHECK: return [[OUT]] : tensor<3x4xf16>
}

}

// -----

// CHECK-LABEL: @FP32Eye
module @FP32Eye {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "Parameter_201" : tensor<1xsi32>
        DataInfo "Parameter_201" : tensor<1xsi32>
    }
    outputsInfo : {
        // CHECK: DataInfo "Eye_202" : tensor<128x128xf32>
        DataInfo "Eye_202" : tensor<128x128xf32>
    }

// CHECK: func.func @main([[ARG0:[^:]+]]: tensor<1xsi32>) -> tensor<128x128xf16>
func.func @main(%arg0: tensor<1xsi32>) -> tensor<128x128xf32> {
    %0 = IE.Eye(%arg0) {num_rows_value = 128 : i64, num_columns_value = 128 : i64, batch_shape_value = [0], outputType = f32, operandSegmentSizes = array<i32: 0, 0, 1, 0>} : tensor<1xsi32>-> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>

    // CHECK:       [[OUT:%.+]] = IE.Eye([[ARG0]])
    // CHECK-SAME:      outputType = f16
    // CHECK-SAME:      tensor<1xsi32> -> tensor<128x128xf16>
    // CHECK: return [[OUT]] : tensor<128x128xf16>
}

}

// -----

// CHECK-LABEL: @FP16Eye
module @FP16Eye {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "Parameter_201" : tensor<1xsi32>
        DataInfo "Parameter_201" : tensor<1xsi32>
    }
    outputsInfo : {
        // CHECK: DataInfo "Eye_202" : tensor<128x128xf32>
        DataInfo "Eye_202" : tensor<128x128xf32>
    }

// CHECK: func.func @main([[ARG0:[^:]+]]: tensor<1xsi32>) -> tensor<128x128xf16>
func.func @main(%arg0: tensor<1xsi32>) -> tensor<128x128xf16> {
    %0 = IE.Eye(%arg0) {num_rows_value = 128 : i64, num_columns_value = 128 : i64, batch_shape_value = [0], outputType = f16, operandSegmentSizes = array<i32: 0, 0, 1, 0>} : tensor<1xsi32>-> tensor<128x128xf16>
    return %0 : tensor<128x128xf16>

    // CHECK:       [[OUT:%.+]] = IE.Eye([[ARG0]])
    // CHECK-SAME:      outputType = f16
    // CHECK-SAME:      tensor<1xsi32> -> tensor<128x128xf16>
    // CHECK: return [[OUT]] : tensor<128x128xf16>
}

}

// -----

!qElemType = !quant.uniform<u8:f32, 2.4627450980392158>
// CHECK: !qElemType = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @FP32Quantize
module @FP32Quantize {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "Parameter_200" : tensor<1x128x256x256xf32>
        DataInfo "Parameter_200" : tensor<1x128x256x256xf32>
        // CHECK: DataInfo "Parameter_201" : tensor<1x128x1x1xf32>
        DataInfo "Parameter_201" : tensor<1x128x1x1xf32>
    }
    outputsInfo : {
        // CHECK: DataInfo "Result_202" : tensor<1x1x256x256xf32>
        DataInfo "Result_202" : tensor<1x1x256x256xf32>
    }

// CHECK: func.func @main([[ARG0:[^:]+]]: tensor<1x128x256x256xf16>, [[ARG1:[^:]+]]: tensor<1x128x1x1xf16>) -> tensor<1x1x256x256xf16>
func.func @main(%arg0: tensor<1x128x256x256xf32>, %arg1: tensor<1x128x1x1xf32>) -> tensor<1x1x256x256xf32> {
    %0 = IE.Quantize(%arg1) {dstElemType = !qElemType} : tensor<1x128x1x1xf32> -> tensor<1x128x1x1x!qElemType>
    %1 = IE.Convolution(%arg0, %0) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x128x256x256xf32>, tensor<1x128x1x1x!qElemType>
            -> tensor<1x1x256x256xf32>
    return %1 : tensor<1x1x256x256xf32>

    // CHECK:       [[QUANT:%.+]] = IE.Quantize([[ARG1]])
    // CHECK-SAME:      dstElemType = !qElemType
    // CHECK-SAME:      tensor<1x128x1x1xf16> -> tensor<1x128x1x1x!qElemType>
    // CHECK:       [[CONV:%.+]] = IE.Convolution([[ARG0]], [[QUANT]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x128x256x256xf16>, tensor<1x128x1x1x!qElemType> -> tensor<1x1x256x256xf16>
}

}

// -----

!qElemType = !quant.uniform<u8:f32, 2.4627450980392158>
// CHECK: !qElemType = !quant.uniform<u8:f16, 2.4627450980392158>

// CHECK-LABEL: @FP32QuantizeCastDequantize
module @FP32QuantizeCastDequantize {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "Parameter_201" : tensor<256x2048xui8>
        DataInfo "Parameter_201" : tensor<256x2048xui8>
    }
    outputsInfo : {
        // CHECK: DataInfo "Result_202" : tensor<256x2048xf32>
        DataInfo "Result_202" : tensor<256x2048xf32>
    }

// CHECK: func.func @main([[ARG0:[^:]+]]: tensor<256x2048xui8>) -> tensor<256x2048xf16>
func.func @main(%arg0: tensor<256x2048xui8>) -> tensor<256x2048xf32> {
    %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} : tensor<256x2048xui8> -> tensor<256x2048x!qElemType>
    %1 = IE.Dequantize(%0) {dstElemType = f32} : tensor<256x2048x!qElemType> -> tensor<256x2048xf32>
    return %1 : tensor<256x2048xf32>

    // CHECK:       [[QUANT:%.+]] = IE.QuantizeCast([[ARG0]])
    // CHECK-SAME:      dstElemType = !qElemType
    // CHECK-SAME:      tensor<256x2048xui8> -> tensor<256x2048x!qElemType>
    // CHECK:       [[DEQUANT:%.+]] = IE.Dequantize([[QUANT]])
    // CHECK-SAME:      dstElemType = f16
    // CHECK-SAME:      tensor<256x2048x!qElemType> -> tensor<256x2048xf16>
    // CHECK: return [[DEQUANT]] : tensor<256x2048xf16>
}

}

// -----

!qElemType = !quant.uniform<u8:f32, 2.4627450980392158>
// CHECK: !qElemType = !quant.uniform<u8:f32, 2.4627450980392158>

// CHECK-LABEL: @NotFP32QuantizeCastDynamicDequantize
module @NotFP32QuantizeCastDynamicDequantize {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "Parameter_201" : tensor<256x2048xui8>
        DataInfo "Parameter_201" : tensor<256x2048xui8>
        // CHECK: DataInfo "Parameter_202" : tensor<1x2048xf32>
        DataInfo "Parameter_202" : tensor<1x2048xf32>
    }
    outputsInfo : {
        // CHECK: DataInfo "Result_202" : tensor<256x2048xf32>
        DataInfo "Result_202" : tensor<256x2048xf32>
    }

// CHECK: func.func @main([[ARG0:[^:]+]]: tensor<256x2048xui8>, [[SCALE:[^:]+]]: tensor<1x2048xf16>) -> tensor<256x2048xf16>
func.func @main(%arg0: tensor<256x2048xui8>, %scale: tensor<1x2048xf32>) -> tensor<256x2048xf32> {
    %0 = IE.QuantizeCast(%arg0) {dstElemType = !qElemType} : tensor<256x2048xui8> -> tensor<256x2048x!qElemType>
    %1 = IE.DynamicDequantize(%0, %scale) {dstElemType = f32} : tensor<256x2048x!qElemType>, tensor<1x2048xf32> -> tensor<256x2048xf32>
    return %1 : tensor<256x2048xf32>

    // CHECK:       [[SCALE_F16:%.+]] = IE.Convert([[SCALE]])
    // CHECK-SAME:      dstElemType = f32
    // CHECK-SAME:      tensor<1x2048xf16> -> tensor<1x2048xf32>

    // CHECK:       [[QUANT:%.+]] = IE.QuantizeCast([[ARG0]])
    // CHECK-SAME:      dstElemType = !qElemType
    // CHECK-SAME:      tensor<256x2048xui8> -> tensor<256x2048x!qElemType>
    // CHECK:       [[DEQUANT:%.+]] = IE.DynamicDequantize([[QUANT]], [[SCALE_F16]])
    // CHECK-SAME:      dstElemType = f32
    // CHECK-SAME:      tensor<256x2048x!qElemType>, tensor<1x2048xf32> -> tensor<256x2048xf32>

    // CHECK:       [[OUT:%.+]] = IE.Convert([[DEQUANT]])
    // CHECK-SAME:      dstElemType = f16
    // CHECK-SAME:      tensor<256x2048xf32> -> tensor<256x2048xf16>
    // CHECK: return [[OUT]] : tensor<256x2048xf16>
}

}

// -----

// CHECK-LABEL: @TwoFunctions
module @TwoFunctions {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        // CHECK: DataInfo "input" : tensor<1x3x62x62xui8>
        DataInfo "input" : tensor<1x3x62x62xui8>
    } outputsInfo : {
        // CHECK: DataInfo "output" : tensor<1x48x60x60xf16>
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @foo1({{[^:]+}}: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
    func.func @foo1(%arg0: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.000000e+00> : tensor<48x3x3x3xf32>
        %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        return %0 : tensor<1x48x60x60xf32>
    }

    // CHECK: func.func @foo2({{[^:]+}}: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    func.func @foo2(%arg0: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %0 = IE.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %0 : tensor<1x48x60x60xf32>
    }

    // CHECK: func.func @main([[ARG0:[^:]+]]: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
    func.func @main(%arg0: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
        %0 = call @foo1(%arg0) : (tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32>
        %1 = call @foo2(%0) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        return %1 : tensor<1x48x60x60xf32>

        // CHECK: [[OUT1:%.+]] = call @foo1([[ARG0]]) : (tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
        // CHECK: [[OUT2:%.+]] = call @foo2([[OUT1]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        // CHECK: return [[OUT2]] : tensor<1x48x60x60xf16>
    }
}

// -----

// CHECK-LABEL: @NotConvertBitWiseOp
module @NotConvertBitWiseOp {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "data" : tensor<1x1024xf16>
        DataInfo "data" : tensor<1x1024xf16>
    }
    outputsInfo : {
        // CHECK: DataInfo "prob" : tensor<1x1024xf16>
        DataInfo "prob" : tensor<1x1024xf16>
    }

// CHECK: func.func @main([[ARG0:[^:]+]]: tensor<1x1024xf16>) -> tensor<1x1024xf16>
func.func @main(%arg0: tensor<1x1024xf16>) -> tensor<1x1024xf16> {
    %0 = IE.Convert(%arg0) {dstElemType = i8} : tensor<1x1024xf16> -> tensor<1x1024xi8>
    %1 = IE.BitwiseNot(%0) : tensor<1x1024xi8> -> tensor<1x1024xi8>
    %2 = IE.BitwiseAnd(%0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1x1024xi8> -> tensor<1x1024xi8>
    %3 = IE.BitwiseOr(%0, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1x1024xi8> -> tensor<1x1024xi8>
    %4 = IE.BitwiseXor(%0, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1x1024xi8> -> tensor<1x1024xi8>
    %5 = IE.BitwiseRightShift(%0, %4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1x1024xi8> -> tensor<1x1024xi8>
    %6 = IE.BitwiseLeftShift(%0, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1x1024xi8> -> tensor<1x1024xi8>
    %7 = IE.Convert(%6) {dstElemType = f16} : tensor<1x1024xi8> -> tensor<1x1024xf16>
    return %7 : tensor<1x1024xf16>
    // CHECK:       [[CONVER:%.+]] = IE.Convert([[ARG0]]) {dstElemType = i8} : tensor<1x1024xf16> -> tensor<1x1024xi8>

    // CHECK:       [[BITWISENOT:%.+]] = IE.BitwiseNot([[CONVER]]) : tensor<1x1024xi8> -> tensor<1x1024xi8>
    // CHECK:       [[BITWISEAND:%.+]] = IE.BitwiseAnd([[CONVER]], [[BITWISENOT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1x1024xi8> -> tensor<1x1024xi8>
    // CHECK:       [[BITWISEOR:%.+]] = IE.BitwiseOr([[CONVER]], [[BITWISEAND]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1x1024xi8> -> tensor<1x1024xi8>
    // CHECK:       [[BITWISEXOR:%.+]] = IE.BitwiseXor([[CONVER]], [[BITWISEOR]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1x1024xi8> -> tensor<1x1024xi8>
    // CHECK:       [[BITWISERIGHTSHIFT:%.+]] = IE.BitwiseRightShift([[CONVER]], [[BITWISEXOR]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1x1024xi8> -> tensor<1x1024xi8>
    // CHECK:       [[BITWISELEFTSHIFT:%.+]] = IE.BitwiseLeftShift([[CONVER]], [[BITWISERIGHTSHIFT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1x1024xi8> -> tensor<1x1024xi8>

    // CHECK:       [[CONVER1:%.+]] = IE.Convert([[BITWISELEFTSHIFT]]) {dstElemType = f16} : tensor<1x1024xi8> -> tensor<1x1024xf16>
    // CHECK:       return [[CONVER1]] : tensor<1x1024xf16>

}

}

// -----

// CHECK-LABEL: @FP32Less
module @FP32Less {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            // CHECK: DataInfo "Input" : tensor<1x2xf16>
            // CHECK: DataInfo "Const" : tensor<1x1xf16>
            DataInfo "Input" : tensor<1x2xf16>
            DataInfo "Const" : tensor<1x1xf16>
        }
        outputsInfo : {
            // CHECK: DataInfo "Out" : tensor<1x2xf16>
            DataInfo "Out" : tensor<1x2xf16>
        }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x2xf16>, [[CST:%.+]]: tensor<1x1xf16>) -> tensor<1x2xf16>
    func.func @main(%input: tensor<1x2xf16>, %cst: tensor<1x1xf16>) -> tensor<1x2xf16> {
        %conver_input = IE.Convert(%input) {dstElemType = f32} : tensor<1x2xf16> -> tensor<1x2xf32>
        %conver_cst = IE.Convert(%cst) {dstElemType = f32} : tensor<1x1xf16> -> tensor<1x1xf32>
        %0 = IE.Less(%conver_input, %conver_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2xf32>, tensor<1x1xf32> -> tensor<1x2xi8>
        %out = IE.Convert(%0) {dstElemType = f16} : tensor<1x2xi8> -> tensor<1x2xf16>
        return %out : tensor<1x2xf16>

        // CHECK: [[OUT:%.+]] = IE.Less([[INPUT]], [[CST]])
        // CHECK-SAME: tensor<1x2xf16>, tensor<1x1xf16> -> tensor<1x2xi8>

    }
}


// -----

// CHECK-LABEL: @FP32GreaterEqual
module @FP32GreaterEqual {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            // CHECK: DataInfo "Input" : tensor<1x2xf16>
            // CHECK: DataInfo "Const" : tensor<1x1xf16>
            DataInfo "Input" : tensor<1x2xf16>
            DataInfo "Const" : tensor<1x1xf16>
        }
        outputsInfo : {
            // CHECK: DataInfo "Out" : tensor<1x2xf16>
            DataInfo "Out" : tensor<1x2xf16>
        }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x2xf16>, [[CST:%.+]]: tensor<1x1xf16>) -> tensor<1x2xf16>
    func.func @main(%input: tensor<1x2xf16>, %cst: tensor<1x1xf16>) -> tensor<1x2xf16> {
        %conver_input = IE.Convert(%input) {dstElemType = f32} : tensor<1x2xf16> -> tensor<1x2xf32>
        %conver_cst = IE.Convert(%cst) {dstElemType = f32} : tensor<1x1xf16> -> tensor<1x1xf32>
        %0 = IE.GreaterEqual(%conver_input, %conver_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2xf32>, tensor<1x1xf32> -> tensor<1x2xf32>
        %out = IE.Convert(%0) {dstElemType = f16} : tensor<1x2xf32> -> tensor<1x2xf16>
        return %out : tensor<1x2xf16>

        // CHECK: [[OUT:%.+]] = IE.GreaterEqual([[INPUT]], [[CST]])
        // CHECK-SAME: tensor<1x2xf16>, tensor<1x1xf16> -> tensor<1x2xf16>
    }
}

// -----

// CHECK-LABEL: @SelectOpDoNotConvertSi32ToFP16
module @SelectOpDoNotConvertSi32ToFP16 {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "flag" : tensor<1x1024xi8>
        // CHECK: DataInfo "data" : tensor<1x1024xsi32>
        DataInfo "flag" : tensor<1x1024xi8>
        DataInfo "data" : tensor<1x1024xsi32>
    }
    outputsInfo : {
        // CHECK: DataInfo "out" : tensor<1x1024xsi32>
        DataInfo "out" : tensor<1x1024xsi32>
    }

// CHECK: @main([[ARG0:[^:]+]]: tensor<1x1024xf16>, [[ARG1:[^:]+]]: tensor<1x1024xsi32>)
// CHECK:  -> tensor<1x1024xsi32>
func.func @main(%arg0: tensor<1x1024xi8>, %arg1: tensor<1x1024xsi32>) -> tensor<1x1024xsi32> {
    %cst = const.Declare tensor<1xsi32> = dense<256> : tensor<1xsi32>
    %0 = IE.Select(%arg0, %cst, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1xsi32>, tensor<1x1024xsi32> -> tensor<1x1024xsi32>
    return %0 : tensor<1x1024xsi32>

    // CHECK:     [[CST:%.+]] = const.Declare tensor<1xsi32> = dense<256> : tensor<1xsi32>
    // CHECK:     [[CONVERT_COND:%.+]] = IE.Convert([[ARG0]]) {dstElemType = i8} : tensor<1x1024xf16> -> tensor<1x1024xi8>
    // CHECK:     [[SELECT:%.+]] = IE.Select([[CONVERT_COND]], [[CST]], [[ARG1]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x1024xi8>, tensor<1xsi32>, tensor<1x1024xsi32> -> tensor<1x1024xsi32>
    // CHECK:     return [[SELECT]] : tensor<1x1024xsi32>
}

}

// -----

// CHECK-LABEL: @SelectSI64NotConvertedToFP16
module @SelectSI64NotConvertedToFP16 {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "input_cond" : tensor<4xsi16>
        DataInfo "input_cond" : tensor<4xsi16>
        // CHECK: DataInfo "input_data" : tensor<4xsi64>
        DataInfo "input_data" : tensor<4xsi64>
    }
    outputsInfo : {
        // CHECK: DataInfo "output" : tensor<4xsi64>
        DataInfo "output" : tensor<4xsi64>
    }

// CHECK: func.func @main([[ARG0:[^:]+]]: tensor<4xsi16>, [[ARG1:[^:]+]]: tensor<4xsi64>) -> tensor<4xsi64>
func.func @main(%arg0 : tensor<4xsi16>, %arg1 : tensor<4xsi64>) -> tensor<4xsi64> {
    %cst = const.Declare tensor<4xsi64> = dense<0> : tensor<4xsi64>
    %0 = IE.Select(%arg0, %arg1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4xsi16>, tensor<4xsi64>, tensor<4xsi64> -> tensor<4xsi64>
    return %0 : tensor<4xsi64>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<4xsi64>
    // CHECK:     IE.Select([[ARG0]], [[ARG1]], [[CST]]) {{.*}} tensor<4xsi16>, tensor<4xsi64>, tensor<4xsi64> -> tensor<4xsi64>
    // CHECK-NOT: IE.Convert
}

}

// -----

// CHECK-LABEL: @SelectOpSi16DoNotConvertToFP16
module @SelectOpSi16DoNotConvertToFP16 {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "flag" : tensor<1x1024xi8>
        // CHECK: DataInfo "data" : tensor<1x1024xsi16>
        DataInfo "flag" : tensor<1x1024xi8>
        DataInfo "data" : tensor<1x1024xsi16>
    }
    outputsInfo : {
        // CHECK: DataInfo "out" : tensor<1x1024xsi16>
        DataInfo "out" : tensor<1x1024xsi16>
    }

// CHECK: @main([[ARG0:[^:]+]]: tensor<1x1024xf16>, [[ARG1:[^:]+]]: tensor<1x1024xsi16>)
// CHECK:  -> tensor<1x1024xsi16>
func.func @main(%arg0: tensor<1x1024xi8>, %arg1: tensor<1x1024xsi16>) -> tensor<1x1024xsi16> {
    %cst = const.Declare tensor<1xsi16> = dense<256> : tensor<1xsi16>
    %0 = IE.Select(%arg0, %cst, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1xsi16>, tensor<1x1024xsi16> -> tensor<1x1024xsi16>
    return %0 : tensor<1x1024xsi16>

    // CHECK:     [[CST:%.+]] = const.Declare tensor<1xsi16> = dense<256> : tensor<1xsi16>
    // CHECK:     [[CONVERT_COND:%.+]] = IE.Convert([[ARG0]]) {dstElemType = i8} : tensor<1x1024xf16> -> tensor<1x1024xi8>
    // CHECK:     [[SELECT:%.+]] = IE.Select([[CONVERT_COND]], [[CST]], [[ARG1]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x1024xi8>, tensor<1xsi16>, tensor<1x1024xsi16> -> tensor<1x1024xsi16>
    // CHECK:     return [[SELECT]] : tensor<1x1024xsi16>
}

}

// -----

// CHECK-LABEL: @SelectOpSi64CondWithF32Data
// Verify that si64 condition is converted to f16 when data operands become f16.
module @SelectOpSi64CondWithF32Data {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "cond" : tensor<1x1024xsi64>
        // CHECK: DataInfo "data" : tensor<1x1024xf32>
        DataInfo "cond" : tensor<1x1024xsi64>
        DataInfo "data" : tensor<1x1024xf32>
    }
    outputsInfo : {
        // CHECK: DataInfo "out" : tensor<1x1024xf32>
        DataInfo "out" : tensor<1x1024xf32>
    }

// CHECK: @main([[ARG0:[^:]+]]: tensor<1x1024xsi64>, [[ARG1:[^:]+]]: tensor<1x1024xf16>)
// CHECK:  -> tensor<1x1024xf16>
func.func @main(%arg0: tensor<1x1024xsi64>, %arg1: tensor<1x1024xf32>) -> tensor<1x1024xf32> {
    %cst = const.Declare tensor<1xf32> = dense<0.0> : tensor<1xf32>
    %0 = IE.Select(%arg0, %arg1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xsi64>, tensor<1x1024xf32>, tensor<1xf32> -> tensor<1x1024xf32>
    return %0 : tensor<1x1024xf32>

    // CHECK-DAG: [[CST:%.+]] = const.Declare tensor<1xf16> = dense<0.000000e+00> : tensor<1xf32>, [#const.CastElemType<f16>]
    // CHECK:     [[CONVERT_COND:%.+]] = IE.Convert([[ARG0]])
    // CHECK-SAME:      {dstElemType = f16} : tensor<1x1024xsi64> -> tensor<1x1024xf16>
    // CHECK:     [[SELECT:%.+]] = IE.Select([[CONVERT_COND]], [[ARG1]], [[CST]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x1024xf16>, tensor<1x1024xf16>, tensor<1xf16> -> tensor<1x1024xf16>
    // CHECK:     return [[SELECT]] : tensor<1x1024xf16>
}

}

// -----

// CHECK-LABEL: @FP32LessEqual
module @FP32LessEqual {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            // CHECK: DataInfo "Input" : tensor<1x2xf16>
            // CHECK: DataInfo "Const" : tensor<1x1xf16>
            DataInfo "Input" : tensor<1x2xf16>
            DataInfo "Const" : tensor<1x1xf16>
        }
        outputsInfo : {
            // CHECK: DataInfo "Out" : tensor<1x2xf16>
            DataInfo "Out" : tensor<1x2xf16>
        }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x2xf16>, [[CST:%.+]]: tensor<1x1xf16>) -> tensor<1x2xf16>
    func.func @main(%input: tensor<1x2xf16>, %cst: tensor<1x1xf16>) -> tensor<1x2xf16> {
        %conver_input = IE.Convert(%input) {dstElemType = f32} : tensor<1x2xf16> -> tensor<1x2xf32>
        %conver_cst = IE.Convert(%cst) {dstElemType = f32} : tensor<1x1xf16> -> tensor<1x1xf32>
        %0 = IE.LessEqual(%conver_input, %conver_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2xf32>, tensor<1x1xf32> -> tensor<1x2xi8>
        %out = IE.Convert(%0) {dstElemType = f16} : tensor<1x2xi8> -> tensor<1x2xf16>
        return %out : tensor<1x2xf16>

        // CHECK: [[OUT:%.+]] = IE.LessEqual([[INPUT]], [[CST]])
        // CHECK-SAME: tensor<1x2xf16>, tensor<1x1xf16> -> tensor<1x2xi8>

    }
}

// -----

// CHECK-LABEL: @FP32Greater
module @FP32Greater {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            // CHECK: DataInfo "Input" : tensor<1x2xf16>
            // CHECK: DataInfo "Const" : tensor<1x1xf16>
            DataInfo "Input" : tensor<1x2xf16>
            DataInfo "Const" : tensor<1x1xf16>
        }
        outputsInfo : {
            // CHECK: DataInfo "Out" : tensor<1x2xf16>
            DataInfo "Out" : tensor<1x2xf16>
        }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x2xf16>, [[CST:%.+]]: tensor<1x1xf16>) -> tensor<1x2xf16>
    func.func @main(%input: tensor<1x2xf16>, %cst: tensor<1x1xf16>) -> tensor<1x2xf16> {
        %conver_input = IE.Convert(%input) {dstElemType = f32} : tensor<1x2xf16> -> tensor<1x2xf32>
        %conver_cst = IE.Convert(%cst) {dstElemType = f32} : tensor<1x1xf16> -> tensor<1x1xf32>
        %0 = IE.Greater(%conver_input, %conver_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2xf32>, tensor<1x1xf32> -> tensor<1x2xi8>
        %out = IE.Convert(%0) {dstElemType = f16} : tensor<1x2xi8> -> tensor<1x2xf16>
        return %out : tensor<1x2xf16>

        // CHECK: [[OUT:%.+]] = IE.Greater([[INPUT]], [[CST]])
        // CHECK-SAME: tensor<1x2xf16>, tensor<1x1xf16> -> tensor<1x2xi8>

    }
}

// -----

// CHECK-LABEL: @fp32DynamicDataMask
module @fp32DynamicDataMask {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            // CHECK: DataInfo "Input" : tensor<4xsi32>
            DataInfo "Input" : tensor<4xsi32>
        }
        outputsInfo : {
            // CHECK: DataInfo "Out" : tensor<1x3x32x32xf32>
            DataInfo "Out" : tensor<1x3x32x32xf32>
        }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<4xsi32>) -> tensor<1x3x32x32xf16> {
    func.func @main(%arg0: tensor<4xsi32>) -> tensor<1x3x32x32xf32> {
        %0 = IE.DynamicDataMask(%arg0) {outputTensorType = tensor<1x3x32x32xf32>} : tensor<4xsi32> -> tensor<1x3x32x32xf32>
        return %0 : tensor<1x3x32x32xf32>

        // CHECK: [[VAR0:%.+]] = IE.DynamicDataMask([[INPUT]]) {outputTensorType = tensor<1x3x32x32xf16>} : tensor<4xsi32> -> tensor<1x3x32x32xf16>
        // CHECK: return [[VAR0]] : tensor<1x3x32x32xf16>
    }
}

// -----

// CHECK-LABEL: @uniquifyAndOpInputsPrecision
module @uniquifyAndOpInputsPrecision {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            // CHECK: DataInfo "Input0" : tensor<1x1x1xsi32>
            // CHECK: DataInfo "Input1" : tensor<1x1x1025xsi32>
            // CHECK: DataInfo "Input2" : tensor<1x1x1025xf16>
            DataInfo "Input0" : tensor<1x1x1xsi32>
            DataInfo "Input1" : tensor<1x1x1025xsi32>
            DataInfo "Input2" : tensor<1x1x1025xf16>
        }
        outputsInfo : {
            // CHECK: DataInfo "Out" : tensor<1x1x1025xf16>
            DataInfo "Out" : tensor<1x1x1025xf16>
        }

    // CHECK: func.func @main([[INPUT0:%.+]]: tensor<1x1x1xsi32>, [[INPUT1:%.+]]: tensor<1x1x1025xsi32>, [[INPUT2:%.+]]: tensor<1x1x1025xf16>) -> tensor<1x1x1025xf16> {
    func.func @main(%arg0: tensor<1x1x1xsi32>, %arg1: tensor<1x1x1025xsi32>, %arg2: tensor<1x1x1025xf16>) -> tensor<1x1x1025xf16> {
        %0 = IE.GreaterEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xsi32>, tensor<1x1x1025xsi32> -> tensor<1x1x1025xsi32>
        %1 = IE.And(%arg2, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1025xf16>, tensor<1x1x1025xsi32> -> tensor<1x1x1025xf16>
        return %1 : tensor<1x1x1025xf16>

        // CHECK: [[VAR0:%.+]] = IE.GreaterEqual([[INPUT0]], [[INPUT1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xsi32>, tensor<1x1x1025xsi32> -> tensor<1x1x1025xsi32>
        // CHECK: [[VAR1:%.+]] = IE.Convert([[VAR0]]) {dstElemType = f16} : tensor<1x1x1025xsi32> -> tensor<1x1x1025xf16>
        // CHECK: [[VAR2:%.+]] = IE.And([[INPUT2]], [[VAR1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1025xf16>, tensor<1x1x1025xf16> -> tensor<1x1x1025xf16>
        // CHECK: return [[VAR2]] : tensor<1x1x1025xf16>
    }
}

// -----

// CHECK-LABEL: @FP64toFP16
module @FP64toFP16 {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            // CHECK: DataInfo "data" : tensor<1x1000xf64>
            DataInfo "data" : tensor<1x1000xf64>
        }
        outputsInfo : {
            // CHECK: DataInfo "prob" : tensor<1x1000xf64>
            DataInfo "prob" : tensor<1x1000xf64>
        }

    // CHECK: func.func @main([[ARG0:[^:]+]]: tensor<1x1000xf16>) -> tensor<1x1000xf16>
    func.func @main(%arg0: tensor<1x1000xf64>) -> tensor<1x1000xf64> {
        %cst = const.Declare tensor<1xf64> = dense<1.0> : tensor<1xf64>
        %out = IE.Divide(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1000xf64>, tensor<1xf64> -> tensor<1x1000xf64>
        return %out : tensor<1x1000xf64>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf64>, [#const.CastElemType<f16>]
        // CHECK:       [[OUT:%.+]] = IE.Divide([[ARG0]], [[CST]])
        // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1000xf16>, tensor<1xf16> -> tensor<1x1000xf16>
        // CHECK:       return [[OUT]] : tensor<1x1000xf16>
    }
}

// -----

// CHECK-LABEL: @NormalizeSubgraphWithoutEpsilon
module @NormalizeSubgraphWithoutEpsilon {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            DataInfo "input" : tensor<1x1x40x64xf16>
        }
        outputsInfo : {
            DataInfo "output" : tensor<1x1x40x64xf16>
        }

    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x1x40x64xf16>)
    func.func @main(%arg0: tensor<1x1x40x64xf16>) -> tensor<1x1x40x64xf16> {
        %divide_cst = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
        %0 = IE.Multiply(%arg0, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf16>, tensor<1x1x40x64xf16> -> tensor<1x1x40x64xf16>
        %1 = IE.ReduceSum(%0) {axes_value = [3], keep_dims} : tensor<1x1x40x64xf16> -> tensor<1x1x40x1xf16>
        %2 = IE.Sqrt(%1) : tensor<1x1x40x1xf16> -> tensor<1x1x40x1xf16>
        // CHECK:  [[MULT:%.+]] = IE.Multiply([[ARG0]], [[ARG0]])
        // CHECK:  [[REDUCE_SUM:%.+]] = IE.ReduceSum([[MULT]])
        // CHECK:  [[CLAMP:%.+]] = IE.Clamp([[REDUCE_SUM]]) {max = 6.550400e+04 : f64, min = 9.9999997473787516E-5 : f64}
        // CHECK:  IE.Sqrt([[CLAMP]])
        %3 = IE.Divide(%divide_cst, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf16>, tensor<1x1x40x1xf16> -> tensor<1x1x40x1xf16>
        %4 = IE.Multiply(%arg0, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x40x64xf16>, tensor<1x1x40x1xf16> -> tensor<1x1x40x64xf16>
        return %4 : tensor<1x1x40x64xf16>
    }
}

// -----

// CHECK-LABEL: @UnfusedNormalizeL2WithoutEpsilon
module @UnfusedNormalizeL2WithoutEpsilon {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            DataInfo "input" : tensor<1x192xf16>
        }
        outputsInfo : {
            DataInfo "output" : tensor<1x192xf16>
        }

    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x192xf16>)
    func.func @main(%arg0: tensor<1x192xf16>) -> tensor<1x192xf16> {
        %0 = IE.ReduceL2(%arg0) {axes_value = [1], keep_dims} : tensor<1x192xf16> -> tensor<1x1xf16>
        %1 = IE.Divide(%arg0, %0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x192xf16>, tensor<1x1xf16> -> tensor<1x192xf16>
        return %1 : tensor<1x192xf16>
        // CHECK:  [[REDUCEL2:%.+]] = IE.ReduceL2([[ARG0]])
        // CHECK:  [[CLAMP:%.+]] = IE.Clamp([[REDUCEL2]]) {max = 6.550400e+04 : f64, min = 9.9999997473787516E-5 : f64}
        // CHECK:  IE.Divide([[ARG0]], [[CLAMP]])
    }
}

// -----

// CHECK-LABEL: @AddSqrtDivide
module @AddSqrtDivide {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            DataInfo "input" : tensor<1x1x1xf16>
        }
        outputsInfo : {
            DataInfo "output" : tensor<1x1x1xf16>
        }

    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x1x1xf16>)
    func.func @main(%arg0: tensor<1x1x1xf16>) -> tensor<1x1x1xf16> {
        %add_cst = const.Declare tensor<1x1x1xf16> = dense<1.0000000E-12> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
        %divide_cst = const.Declare tensor<1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
        %0 = IE.Add(%arg0, %add_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf16>, tensor<1x1x1xf16> -> tensor<1x1x1xf16>
        %1 = IE.Sqrt(%0) : tensor<1x1x1xf16> -> tensor<1x1x1xf16>
        %2 = IE.Divide(%divide_cst, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf16>, tensor<1x1x1xf16> -> tensor<1x1x1xf16>
        return %2 : tensor<1x1x1xf16>
    }
    // CHECK:  [[ADD_CST:%.+]] = const.Declare tensor<1x1x1xf16> = dense<9.99999996E-13> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:  [[DIVIDE_CST:%.+]] = const.Declare tensor<1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:  [[ADD:%.+]] = IE.Add([[ARG0]], [[ADD_CST]])
    // CHECK:  [[CLAMP:%.+]] = IE.Clamp([[ADD]]) {max = 6.550400e+04 : f64, min = 9.9999997473787516E-5 : f64}
    // CHECK:  [[SQRT:%.+]] = IE.Sqrt([[CLAMP]])
    // CHECK:  IE.Divide([[DIVIDE_CST]], [[SQRT]])
}

// -----

// CHECK-LABEL: @MaximumSqrtDivide
module @MaximumSqrtDivide {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            DataInfo "input" : tensor<1x1x3072xf16>
        }
        outputsInfo : {
            DataInfo "output" : tensor<1x1x3072xf16>
        }

    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x1x3072xf16>)
    func.func @main(%arg0: tensor<1x1x3072xf16>) -> tensor<1x1x3072xf16> {
        %epsilon_cst = const.Declare tensor<1x1x1xf16> = dense<1.0000000E-12> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
        %power_cst = const.Declare tensor<1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
        %divide_cst = const.Declare tensor<1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
        %multiply_cst = const.Declare tensor<1x1x3072xf16> = dense<1.000000e+00> : tensor<1x1x3072xf32>, [#const.CastElemType<f16>]
        %0 = IE.Power(%arg0, %power_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf16>, tensor<1x1x1xf16> -> tensor<1x1x3072xf16>
        %1 = IE.ReduceMean(%0) {axes_value = [2], keep_dims} : tensor<1x1x3072xf16> -> tensor<1x1x1xf16>
        %2 = IE.Maximum(%1, %epsilon_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf16>, tensor<1x1x1xf16> -> tensor<1x1x1xf16>
        %3 = IE.Sqrt(%2) : tensor<1x1x1xf16> -> tensor<1x1x1xf16>
        %4 = IE.Divide(%divide_cst, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf16>, tensor<1x1x1xf16> -> tensor<1x1x1xf16>
        %5 = IE.Multiply(%arg0, %4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf16>, tensor<1x1x1xf16> -> tensor<1x1x3072xf16>
        %6 = IE.Multiply(%5, %multiply_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3072xf16>, tensor<1x1x3072xf16> -> tensor<1x1x3072xf16>
        return %6 : tensor<1x1x3072xf16>
    }
    // CHECK: [[EPSILON_CST:%.+]] = const.Declare tensor<1x1x1xf16> = dense<9.99999996E-13> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK: [[POWER_CST:%.+]] = const.Declare tensor<1x1x1xf16> = dense<2.000000e+00> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK: [[DIVIDE_CST:%.+]] = const.Declare tensor<1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK: [[POWER:%.+]] = IE.Power([[ARG0]], [[POWER_CST]])
    // CHECK: [[REDUCE_MEAN:%.+]] = IE.ReduceMean([[POWER]])
    // CHECK: [[MAXIMUM:%.+]] = IE.Maximum([[REDUCE_MEAN]], [[EPSILON_CST]])
    // CHECK: [[CLAMP:%.+]] = IE.Clamp([[MAXIMUM]]) {max = 6.550400e+04 : f64, min = 9.9999997473787516E-5 : f64}
    // CHECK: [[SQRT:%.+]] = IE.Sqrt([[CLAMP]])
    // CHECK: [[DIVIDE:%.+]] = IE.Divide([[DIVIDE_CST]], [[SQRT]])
    // CHECK: IE.Multiply([[ARG0]], [[DIVIDE]])
}

// -----

// CHECK-LABEL: @ClampSqrtDivide
module @ClampSqrtDivide {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            DataInfo "input" : tensor<1x1x40x1xf16>
        }
        outputsInfo : {
            DataInfo "output" : tensor<1x1x40x1xf16>
        }

    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x1x40x1xf16>)
    func.func @main(%arg0: tensor<1x1x40x1xf16>) -> tensor<1x1x40x1xf16> {
        %divide_cst = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
        %0 = IE.Clamp(%arg0) {max = 6.550400e+04 : f64, min = 9.999999960041972E-13 : f64} : tensor<1x1x40x1xf16> -> tensor<1x1x40x1xf16>
        %1 = IE.Sqrt(%0) : tensor<1x1x40x1xf16> -> tensor<1x1x40x1xf16>
        %2 = IE.Divide(%divide_cst, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf16>, tensor<1x1x40x1xf16> -> tensor<1x1x40x1xf16>
        return %2 : tensor<1x1x40x1xf16>
    }
    // CHECK:  [[DIVIDE_CST_CST:%.+]] = const.Declare tensor<1x1x1x1xf16> = dense<1.000000e+00> : tensor<1x1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:  [[CLAMP:%.+]] = IE.Clamp([[ARG0]]) {max = 6.550400e+04 : f64, min = 9.9999997473787516E-5 : f64}
    // CHECK:  [[SQRT:%.+]] = IE.Sqrt([[CLAMP]])
    // CHECK:  IE.Divide([[DIVIDE_CST_CST]], [[SQRT]])
}

// -----

// CHECK-LABEL: @ClampDivide
module @ClampDivide {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            DataInfo "input" : tensor<1x192xf16>
        }
        outputsInfo : {
            DataInfo "output" : tensor<1x192xf16>
        }

    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x192xf16>)
    func.func @main(%arg0: tensor<1x192xf16>) -> tensor<1x192xf16> {
        %0 = IE.ReduceL2(%arg0) {axes_value = [1], keep_dims} : tensor<1x192xf16> -> tensor<1x1xf16>
        %1 = IE.Clamp(%0) {max = 1.7976931348623157E+308 : f64, min = 9.999999960041972E-13 : f64} : tensor<1x1xf16> -> tensor<1x1xf16>
        %2 = IE.Divide(%arg0, %1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x192xf16>, tensor<1x1xf16> -> tensor<1x192xf16>
        return %2 : tensor<1x192xf16>
    }
    // CHECK:  [[REDUCEL2:%.+]] = IE.ReduceL2([[ARG0]])
    // CHECK:  [[CLAMP:%.+]] = IE.Clamp([[REDUCEL2]]) {max = 6.550400e+04 : f64, min = 9.9999997473787516E-5 : f64}
    // CHECK:  IE.Divide([[ARG0]], [[CLAMP]])
}

// -----

// CHECK-LABEL: @AddPowerNeg0p5
module @AddPowerNeg0p5 {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            DataInfo "input" : tensor<1x1x1xf16>
        }
        outputsInfo : {
            DataInfo "output" : tensor<1x1x1xf16>
        }

    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x1x1xf16>)
    func.func @main(%arg0: tensor<1x1x1xf16>) -> tensor<1x1x1xf16> {
        %power_cst = const.Declare tensor<1x1x1xf16> = dense<-0.5> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
        %add_cst = const.Declare tensor<1x1x1xf16> = dense<1.587e-11> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
        %0 = IE.Add(%arg0, %add_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf16>, tensor<1x1x1xf16> -> tensor<1x1x1xf16>
        %1 = IE.Power(%0, %power_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf16>, tensor<1x1x1xf16> -> tensor<1x1x1xf16>
        return %1 : tensor<1x1x1xf16>
    }
    // CHECK:  [[POWER_CST:%.+]] = const.Declare tensor<1x1x1xf16> = dense<-5.000000e-01> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:  [[ADD_CST:%.+]] = const.Declare tensor<1x1x1xf16> = dense<1.587000e-11> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:  [[ADD:%.+]] = IE.Add([[ARG0]], [[ADD_CST]])
    // CHECK:  [[CLAMP:%.+]] = IE.Clamp([[ADD]]) {max = 6.550400e+04 : f64, min = 9.9999997473787516E-5 : f64}
    // CHECK:  IE.Power([[CLAMP]], [[POWER_CST]])
}

// -----

// CHECK-LABEL: @AddPowerNeg1p0
module @AddPowerNeg1p0 {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            DataInfo "input" : tensor<1x1x1xf16>
        }
        outputsInfo : {
            DataInfo "output" : tensor<1x1x1xf16>
        }

    // CHECK:  func.func @main([[ARG0:%.+]]: tensor<1x1x1xf16>)
    func.func @main(%arg0: tensor<1x1x1xf16>) -> tensor<1x1x1xf16> {
        %power_cst = const.Declare tensor<1x1x1xf16> = dense<-1.0> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
        %add_cst = const.Declare tensor<1x1x1xf16> = dense<1.587e-11> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
        %0 = IE.Add(%arg0, %add_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf16>, tensor<1x1x1xf16> -> tensor<1x1x1xf16>
        %1 = IE.Power(%0, %power_cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf16>, tensor<1x1x1xf16> -> tensor<1x1x1xf16>
        return %1 : tensor<1x1x1xf16>
    }
    // CHECK:  [[POWER_CST:%.+]] = const.Declare tensor<1x1x1xf16> = dense<-1.000000e+00> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:  [[ADD_CST:%.+]] = const.Declare tensor<1x1x1xf16> = dense<1.587000e-11> : tensor<1x1x1xf32>, [#const.CastElemType<f16>]
    // CHECK:  [[ADD:%.+]] = IE.Add([[ARG0]], [[ADD_CST]])
    // CHECK:  [[CLAMP:%.+]] = IE.Clamp([[ADD]]) {max = 6.550400e+04 : f64, min = 9.9999997473787516E-5 : f64}
    // CHECK:  IE.Power([[CLAMP]], [[POWER_CST]])
}

// -----

// CHECK-LABEL: @SkipClampForDivideSelect
module @SkipClampForDivideSelect {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            DataInfo "input1" : tensor<150x1x1xsi64>
            DataInfo "input2" : tensor<150x1x768xf32>
        }
        outputsInfo : {
            DataInfo "output" : tensor<1x768xf32>
        }

    func.func @main(%arg0 : tensor<150x1x1xsi64>, %arg1 : tensor<150x1x768xf32>) -> tensor<1x768xf32> {
        %cst_greater = const.Declare tensor<1x1xsi64> = dense<0> : tensor<1x1xsi64>
        %cst_select = const.Declare tensor<1x768xf32> = dense<0.000000e+00> : tensor<1x768xf32>
        %0 = IE.ReduceSum(%arg0) {axes_value = [0]} : tensor<150x1x1xsi64> -> tensor<1x1xsi64>
        %1 = IE.Greater(%0, %cst_greater) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1xsi64>, tensor<1x1xsi64> -> tensor<1x1xi8>
        %2 = IE.ReduceSum(%arg1) {axes_value = [0]} : tensor<150x1x768xf32> -> tensor<1x768xf32>
        %3 = IE.Convert(%0) {dstElemType = f32} : tensor<1x1xsi64> -> tensor<1x1xf32>
        %4 = IE.Divide(%2, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x768xf32>, tensor<1x1xf32> -> tensor<1x768xf32>
        %5 = IE.Select(%1, %4, %cst_select) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1xi8>, tensor<1x768xf32>, tensor<1x768xf32> -> tensor<1x768xf32>
        return %5 : tensor<1x768xf32>
        // CHECK-NOT:  IE.Clamp
    }
}

// -----

// CHECK-LABEL: @SkipClampForDivideConvertConvertSelect
module @SkipClampForDivideConvertConvertSelect {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            DataInfo "input1" : tensor<150x1x1xsi64>
            DataInfo "input2" : tensor<150x1x768xf32>
        }
        outputsInfo : {
            DataInfo "output" : tensor<1x768xf32>
        }

    func.func @main(%arg0 : tensor<150x1x1xsi64>, %arg1 : tensor<150x1x768xf32>) -> tensor<1x768xf32> {
        %cst_greater = const.Declare tensor<1x1xsi64> = dense<0> : tensor<1x1xsi64>
        %cst_select = const.Declare tensor<1x768xf32> = dense<0.000000e+00> : tensor<1x768xf32>
        %0 = IE.ReduceSum(%arg0) {axes_value = [0]} : tensor<150x1x1xsi64> -> tensor<1x1xsi64>
        %1 = IE.Greater(%0, %cst_greater) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1xsi64>, tensor<1x1xsi64> -> tensor<1x1xi8>
        %2 = IE.ReduceSum(%arg1) {axes_value = [0]} : tensor<150x1x768xf32> -> tensor<1x768xf32>
        %3 = IE.Convert(%0) {dstElemType = f32} : tensor<1x1xsi64> -> tensor<1x1xf32>
        %4 = IE.Divide(%2, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x768xf32>, tensor<1x1xf32> -> tensor<1x768xf32>
        %5 = IE.Convert(%4) {dstElemType = f16} : tensor<1x768xf32> -> tensor<1x768xf16>
        %6 = IE.Convert(%5) {dstElemType = f32} : tensor<1x768xf16> -> tensor<1x768xf32>
        %7 = IE.Select(%1, %6, %cst_select) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1xi8>, tensor<1x768xf32>, tensor<1x768xf32> -> tensor<1x768xf32>
        return %7 : tensor<1x768xf32>
        // CHECK-NOT:  IE.Clamp
    }
}

// -----

// CHECK-LABEL: @keepFP32
module @keepFP32 {
    net.PrecisionRequirement precisionInfo : {
        net.PrecisionInfo "op1" "IE.Multiply"
    }

    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            // CHECK: DataInfo "data" : tensor<1x1000xf32>
            DataInfo "data" : tensor<1x1000xf32>
        }
        outputsInfo : {
            // CHECK: DataInfo "prob" : tensor<1x1000xf32>
            DataInfo "prob" : tensor<1x1000xf32>
        }

    // CHECK: func.func @main([[ARG0:[^:]+]]: tensor<1x1000xf16>) -> tensor<1x1000xf16>
    func.func @main(%arg0: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
        %cst = const.Declare tensor<1xf32> = dense<2.0> : tensor<1xf32>
        %out = IE.Multiply(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1000xf32>, tensor<1xf32> -> tensor<1x1000xf32> loc(fused<{name = "op1", type = "Multiply"}>["op1"])
        return %out : tensor<1x1000xf32>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<1xf32> = dense<2.000000e+00> : tensor<1xf32>
		// CHECK:       [[CONVERT_IN:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f32} : tensor<1x1000xf16> -> tensor<1x1000xf32>
        // CHECK:       [[OUT:%.+]] = IE.Multiply([[CONVERT_IN]], [[CST]])
        // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1000xf32>, tensor<1xf32> -> tensor<1x1000xf32>
		// CHECK:       [[CONVERT_OUT:%.+]] = IE.Convert([[OUT]]) {dstElemType = f16} : tensor<1x1000xf32> -> tensor<1x1000xf16>
        // CHECK:       return [[CONVERT_OUT]] : tensor<1x1000xf16>
    }
}

// -----

// CHECK-LABEL: @keepFP32OnlyPrecisionSensitiveOps
module @keepFP32OnlyPrecisionSensitiveOps {
    net.PrecisionRequirement precisionInfo : {
        net.PrecisionInfo "op1" "IE.Multiply"
        net.PrecisionInfo "op2" "IE.Power"
    }

    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            // CHECK: DataInfo "data" : tensor<1x1000xf32>
            DataInfo "data" : tensor<1x1000xf32>
        }
        outputsInfo : {
            // CHECK: DataInfo "prob" : tensor<1x1000xf32>
            DataInfo "prob" : tensor<1x1000xf32>
        }

    // CHECK: func.func @main([[ARG0:[^:]+]]: tensor<1x1000xf16>) -> tensor<1x1000xf16>
    func.func @main(%arg0: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
        %cst = const.Declare tensor<1xf32> = dense<2.0> : tensor<1xf32>
        %0 = IE.Power(%arg0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1000xf32>, tensor<1xf32> -> tensor<1x1000xf32> loc(fused<{name = "op2", type = "Power"}>["op2"])
        %1 = IE.Add(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1000xf32>, tensor<1xf32> -> tensor<1x1000xf32>
        %2 = IE.Multiply(%1, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1000xf32>, tensor<1xf32> -> tensor<1x1000xf32> loc(fused<{name = "op1", type = "Multiply"}>["op1"])
        %out = IE.Divide(%cst, %2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf32>, tensor<1x1000xf32> -> tensor<1x1000xf32>
        return %out : tensor<1x1000xf32>

        // CHECK:       [[CST16:%.+]] = const.Declare tensor<1xf16> = dense<2.000000e+00> : tensor<1xf32>, [#const.CastElemType<f16>]
		// CHECK:       [[CST32:%.+]] = const.Declare tensor<1xf32> = dense<2.000000e+00> : tensor<1xf32>
		// CHECK:       [[CONVERT_IN:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f32} : tensor<1x1000xf16> -> tensor<1x1000xf32>
		// CHECK:       [[POWER:%.+]] = IE.Power([[CONVERT_IN]], [[CST32]])
        // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1000xf32>, tensor<1xf32> -> tensor<1x1000xf32>
		// CHECK:       [[CONVERT_POWER:%.+]] = IE.Convert([[POWER]]) {dstElemType = f16} : tensor<1x1000xf32> -> tensor<1x1000xf16>
        // CHECK:       [[ADD:%.+]] = IE.Add([[CONVERT_POWER]], [[CST16]])
        // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1000xf16>, tensor<1xf16> -> tensor<1x1000xf16>
        // CHECK:       [[CONVERT_ADD:%.+]] = IE.Convert([[ADD]]) {dstElemType = f32} : tensor<1x1000xf16> -> tensor<1x1000xf32>
		// CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[CONVERT_ADD]], [[CST32]])
        // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1000xf32>, tensor<1xf32> -> tensor<1x1000xf32>
        // CHECK:       [[CONVERT_MULTIPLY:%.+]] = IE.Convert([[MULTIPLY]]) {dstElemType = f16} : tensor<1x1000xf32> -> tensor<1x1000xf16>
		// CHECK:       [[CLAMP:%.+]] = IE.Clamp([[CONVERT_MULTIPLY]]) {max = 6.550400e+04 : f64, min = 9.9999997473787516E-5 : f64} : tensor<1x1000xf16> -> tensor<1x1000xf16>
		// CHECK:       [[OUT:%.+]] = IE.Divide([[CST16]], [[CLAMP]])
        // CHECK-SAME:  {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf16>, tensor<1x1000xf16> -> tensor<1x1000xf16>
        // CHECK:       return [[OUT]] : tensor<1x1000xf16>
    }
}

// -----

// CHECK-LABEL: @NotConvertI64ClampToFP16
module @NotConvertI64ClampToFP16 {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            // CHECK: DataInfo "input" : tensor<6x2xsi64>
            DataInfo "input" : tensor<6x2xsi64>
        }
        outputsInfo : {
            // CHECK: DataInfo "output" : tensor<6x2x512xf32>
            DataInfo "output" : tensor<6x2x512xf32>
        }

    // CHECK: func.func @main([[ARG0:[^:]+]]: tensor<6x2xsi64>) -> tensor<6x2x512xf16>
    func.func @main(%arg0: tensor<6x2xsi64>) -> tensor<6x2x512xf32> {
        %cst = const.Declare tensor<12068x512xf32> = dense<1.0> : tensor<12068x512xf32>
        %0 = IE.Clamp(%arg0) {max = 9.2233720368547758E+18 : f64, min = 0.000000e+00 : f64} : tensor<6x2xsi64> -> tensor<6x2xsi64>
        %1 = IE.Gather(%cst, %0) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<12068x512xf32>, tensor<6x2xsi64> -> tensor<6x2x512xf32>

        return %1 : tensor<6x2x512xf32>

        // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<12068x512xf16> = dense<1.000000e+00> : tensor<12068x512xf32>, [#const.CastElemType<f16>]
        // CHECK:       [[CLAMP:%.+]] = IE.Clamp([[ARG0]]) {max = 9.2233720368547758E+18 : f64, min = 0.000000e+00 : f64} : tensor<6x2xsi64> -> tensor<6x2xsi64>
        // CHECK:       [[GATHER:%.+]] = IE.Gather([[CST]], [[CLAMP]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 2 : i64} : tensor<12068x512xf16>, tensor<6x2xsi64> -> tensor<6x2x512xf16>

        // CHECK:       return [[GATHER]] : tensor<6x2x512xf16>
    }
}

// -----

// CHECK-LABEL: @NotConvertI64ClampToFP16WithAffineReshape
module @NotConvertI64ClampToFP16WithAffineReshape {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            // CHECK: DataInfo "input" : tensor<6x2xsi64>
            DataInfo "input" : tensor<6x2xsi64>
        }
        outputsInfo : {
            // CHECK: DataInfo "output" : tensor<12x512xf32>
            DataInfo "output" : tensor<12x512xf32>
        }

    // CHECK: func.func @main([[ARG0:[^:]+]]: tensor<6x2xsi64>) -> tensor<12x512xf16>
    func.func @main(%arg0: tensor<6x2xsi64>) -> tensor<12x512xf32> {
        %cst = const.Declare tensor<12068x512xf32> = dense<1.0> : tensor<12068x512xf32>
        %0 = IE.Clamp(%arg0) {max = 9.2233720368547758E+18 : f64, min = 0.000000e+00 : f64} : tensor<6x2xsi64> -> tensor<6x2xsi64>
        %1 = IE.AffineReshape(%0) {dim_mapping = [[0], [0]], shape_value = [12]} : tensor<6x2xsi64> -> tensor<12xsi64>
        %2 = IE.Gather(%cst, %1) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<12068x512xf32>, tensor<12xsi64> -> tensor<12x512xf32>

        return %2 : tensor<12x512xf32>

        // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<12068x512xf16> = dense<1.000000e+00> : tensor<12068x512xf32>, [#const.CastElemType<f16>]
        // CHECK:       [[CLAMP:%.+]] = IE.Clamp([[ARG0]]) {max = 9.2233720368547758E+18 : f64, min = 0.000000e+00 : f64} : tensor<6x2xsi64> -> tensor<6x2xsi64>
        // CHECK:       [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[CLAMP]])
        // CHECK:       [[GATHER:%.+]] = IE.Gather([[CST]], [[AFFINE_RESHAPE]]) {axis_value = 0 : i64, batch_dims = 0 : i64, indices_rank = 1 : i64} : tensor<12068x512xf16>, tensor<12xsi64> -> tensor<12x512xf16>

        // CHECK:       return [[GATHER]] : tensor<12x512xf16>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @InterpolateInputF32ScaleF32
module @InterpolateInputF32ScaleF32 {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            DataInfo "data" : tensor<1x3x4x6xf32>
            DataInfo "scales" : tensor<2xf32>
        }
        outputsInfo : {
            DataInfo "output" : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        }

    // CHECK:       func.func @main(
    // CHECK-SAME:      [[DATA:%[^:]+]]: tensor<1x3x4x6xf16>,
    // CHECK-SAME:      [[SCALES:%[^:]+]]: tensor<2xf32>
    // CHECK-SAME:    ) -> tensor<1x3x?x?xf16
    func.func @main(%data: tensor<1x3x4x6xf32>, %scales: tensor<2xf32>)
            -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}> {
        %interp = IE.Interpolate(%data, %scales) {
            attr = #IE.Interpolate<mode = <LINEAR>,
                                   shape_calc_mode = <SCALES>,
                                   coord_mode = <HALF_PIXEL>,
                                   nearest_mode = <FLOOR>,
                                   antialias = false,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   cube_coeff = -7.500000e-01 : f64>,
            axes_attr = [2, 3],
            operandSegmentSizes = array<i32: 1, 0, 1, 0>,
            sizes_attr = []}
            : tensor<1x3x4x6xf32>, tensor<2xf32>
                -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        return %interp : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>

        // CHECK-NOT:   IE.Convert({{.*}}) {dstElemType = f16} : tensor<2xf32>
        // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[DATA]], [[SCALES]])
        // CHECK-SAME:    : tensor<1x3x4x6xf16>, tensor<2xf32>
        // CHECK-SAME:        -> tensor<1x3x?x?xf16
        // CHECK:       return [[INTERP]]
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @InterpolateInputF16ScaleF32
module @InterpolateInputF16ScaleF32 {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            DataInfo "data" : tensor<1x3x4x6xf16>
            DataInfo "scales" : tensor<2xf32>
        }
        outputsInfo : {
            DataInfo "output" : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        }

    // CHECK:       func.func @main(
    // CHECK-SAME:      [[DATA:%[^:]+]]: tensor<1x3x4x6xf16>,
    // CHECK-SAME:      [[SCALES:%[^:]+]]: tensor<2xf32>
    // CHECK-SAME:    ) -> tensor<1x3x?x?xf16
    func.func @main(%data: tensor<1x3x4x6xf16>, %scales: tensor<2xf32>)
            -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}> {
        %interp = IE.Interpolate(%data, %scales) {
            attr = #IE.Interpolate<mode = <LINEAR>,
                                   shape_calc_mode = <SCALES>,
                                   coord_mode = <HALF_PIXEL>,
                                   nearest_mode = <FLOOR>,
                                   antialias = false,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   cube_coeff = -7.500000e-01 : f64>,
            axes_attr = [2, 3],
            operandSegmentSizes = array<i32: 1, 0, 1, 0>,
            sizes_attr = []}
            : tensor<1x3x4x6xf16>, tensor<2xf32>
                -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        return %interp : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>

        // CHECK-NOT:   IE.Convert({{.*}}) {dstElemType = f16} : tensor<2xf32>
        // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[DATA]], [[SCALES]])
        // CHECK-SAME:    : tensor<1x3x4x6xf16>, tensor<2xf32>
        // CHECK-SAME:        -> tensor<1x3x?x?xf16
        // CHECK:       return [[INTERP]]
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @InterpolateInputF16ScaleF16
module @InterpolateInputF16ScaleF16 {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            DataInfo "data" : tensor<1x3x4x6xf16>
            DataInfo "scales" : tensor<2xf16>
        }
        outputsInfo : {
            DataInfo "output" : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        }

    // CHECK:       func.func @main(
    // CHECK-SAME:      [[DATA:%[^:]+]]: tensor<1x3x4x6xf16>,
    // CHECK-SAME:      [[SCALES:%[^:]+]]: tensor<2xf16>
    // CHECK-SAME:    ) -> tensor<1x3x?x?xf16
    func.func @main(%data: tensor<1x3x4x6xf16>, %scales: tensor<2xf16>)
            -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}> {
        %interp = IE.Interpolate(%data, %scales) {
            attr = #IE.Interpolate<mode = <LINEAR>,
                                   shape_calc_mode = <SCALES>,
                                   coord_mode = <HALF_PIXEL>,
                                   nearest_mode = <FLOOR>,
                                   antialias = false,
                                   pads_begin = [0, 0, 0, 0],
                                   pads_end = [0, 0, 0, 0],
                                   cube_coeff = -7.500000e-01 : f64>,
            axes_attr = [2, 3],
            operandSegmentSizes = array<i32: 1, 0, 1, 0>,
            sizes_attr = []}
            : tensor<1x3x4x6xf16>, tensor<2xf16>
                -> tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        return %interp : tensor<1x3x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>

        // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[DATA]], [[SCALES]])
        // CHECK-SAME:    : tensor<1x3x4x6xf16>, tensor<2xf16>
        // CHECK-SAME:        -> tensor<1x3x?x?xf16
        // CHECK:       return [[INTERP]]
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SharedScalesAcrossTwoSAPInterpolates
module @SharedScalesAcrossTwoSAPInterpolates {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            DataInfo "data0" : tensor<1x3x4x6xf32>
            DataInfo "data1" : tensor<1x3x4x6xf32>
            DataInfo "scales" : tensor<2xf32>
        }
        outputsInfo : {
            DataInfo "out0" : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
            DataInfo "out1" : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        }

    // CHECK:       func.func @main(
    // CHECK-SAME:      [[D0:%[^:]+]]: tensor<1x3x4x6xf16>,
    // CHECK-SAME:      [[D1:%[^:]+]]: tensor<1x3x4x6xf16>,
    // CHECK-SAME:      [[SCALES:%[^:]+]]: tensor<2xf32>
    func.func @main(%d0: tensor<1x3x4x6xf32>, %d1: tensor<1x3x4x6xf32>, %scales: tensor<2xf32>)
            -> (tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>,
                tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>) {
        %i0 = IE.Interpolate(%d0, %scales) {
            attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>,
                                   nearest_mode = <FLOOR>, antialias = false,
                                   pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
                                   cube_coeff = -7.500000e-01 : f64>,
            axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 1, 0>, sizes_attr = []}
            : tensor<1x3x4x6xf32>, tensor<2xf32>
                -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        %i1 = IE.Interpolate(%d1, %scales) {
            attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>,
                                   nearest_mode = <FLOOR>, antialias = false,
                                   pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
                                   cube_coeff = -7.500000e-01 : f64>,
            axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 1, 0>, sizes_attr = []}
            : tensor<1x3x4x6xf32>, tensor<2xf32>
                -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        return %i0, %i1
            : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>,
              tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>

        // CHECK-NOT:   IE.Convert({{.*}}) {dstElemType = f16} : tensor<2xf32>
        // CHECK:       [[I0:%.+]] = IE.Interpolate([[D0]], [[SCALES]])
        // CHECK-SAME:    : tensor<1x3x4x6xf16>, tensor<2xf32>
        // CHECK:       [[I1:%.+]] = IE.Interpolate([[D1]], [[SCALES]])
        // CHECK-SAME:    : tensor<1x3x4x6xf16>, tensor<2xf32>
        // CHECK:       return [[I0]], [[I1]]
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ScalesSharedWithMultiplyNotPreserved
// The scales input feeds a SAP Interpolate and a Multiply. The non-Interpolate
// user blocks f32 preservation, so the scales is converted to f16 as usual.
module @ScalesSharedWithMultiplyNotPreserved {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            DataInfo "data" : tensor<1x3x4x6xf32>
            DataInfo "scales" : tensor<2xf32>
        }
        outputsInfo : {
            DataInfo "interp" : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
            DataInfo "scaled" : tensor<2xf32>
        }

    // CHECK:       func.func @main(
    // CHECK-SAME:      [[DATA:%[^:]+]]: tensor<1x3x4x6xf16>,
    // CHECK-SAME:      [[SCALES:%[^:]+]]: tensor<2xf16>
    func.func @main(%data: tensor<1x3x4x6xf32>, %scales: tensor<2xf32>)
            -> (tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>,
                tensor<2xf32>) {
        %interp = IE.Interpolate(%data, %scales) {
            attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>,
                                   nearest_mode = <FLOOR>, antialias = false,
                                   pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
                                   cube_coeff = -7.500000e-01 : f64>,
            axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 1, 0>, sizes_attr = []}
            : tensor<1x3x4x6xf32>, tensor<2xf32>
                -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        %scaled = IE.Multiply(%scales, %scales) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
        return %interp, %scaled
            : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>,
              tensor<2xf32>

        // Scales is narrowed to f16 (not preserved) because of the Multiply user.
        // Scale remain fp32 with other user support TODO: E#221311
        // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[DATA]], [[SCALES]])
        // CHECK-SAME:    : tensor<1x3x4x6xf16>, tensor<2xf16>
        // CHECK-SAME:        -> tensor<1x3x?x?xf16
        // CHECK:       [[SCALED:%.+]] = IE.Multiply([[SCALES]], [[SCALES]])
        // CHECK-SAME:    : tensor<2xf16>, tensor<2xf16> -> tensor<2xf16>
        // CHECK:       return [[INTERP]], [[SCALED]]
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @InterpolateInCalleePreservesScales
// The f32 scales network input is forwarded through a func.call into a callee
// (@foo) that holds the SAP Interpolate. The whole forwarding chain is kept at
// f32: the @main argument, the call operand, the @foo argument, and the
// Interpolate scales operand. This mirrors the post-outliner layout where the
// body is outlined out of the entry function.
module @InterpolateInCalleePreservesScales {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            DataInfo "data" : tensor<1x3x4x6xf32>
            DataInfo "scales" : tensor<2xf32>
        }
        outputsInfo : {
            DataInfo "output" : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        }

    // CHECK:       func.func @foo(
    // CHECK-SAME:      [[FOO_DATA:%[^:]+]]: tensor<1x3x4x6xf16>,
    // CHECK-SAME:      [[FOO_SCALES:%[^:]+]]: tensor<2xf32>
    // CHECK-SAME:    ) -> tensor<1x3x?x?xf16
    func.func @foo(%data: tensor<1x3x4x6xf32>, %scales: tensor<2xf32>)
            -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}> {
        %interp = IE.Interpolate(%data, %scales) {
            attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>,
                                   nearest_mode = <FLOOR>, antialias = false,
                                   pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
                                   cube_coeff = -7.500000e-01 : f64>,
            axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 1, 0>, sizes_attr = []}
            : tensor<1x3x4x6xf32>, tensor<2xf32>
                -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        return %interp : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>

        // Scales stays f32 inside the callee; no Convert is inserted for it.
        // CHECK-NOT:   IE.Convert({{.*}}) {dstElemType = f16} : tensor<2xf32>
        // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[FOO_DATA]], [[FOO_SCALES]])
        // CHECK-SAME:    : tensor<1x3x4x6xf16>, tensor<2xf32>
        // CHECK-SAME:        -> tensor<1x3x?x?xf16
        // CHECK:       return [[INTERP]]
    }

    // CHECK:       func.func @main(
    // CHECK-SAME:      [[DATA:%[^:]+]]: tensor<1x3x4x6xf16>,
    // CHECK-SAME:      [[SCALES:%[^:]+]]: tensor<2xf32>
    // CHECK-SAME:    ) -> tensor<1x3x?x?xf16
    func.func @main(%data: tensor<1x3x4x6xf32>, %scales: tensor<2xf32>)
            -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}> {
        %result = func.call @foo(%data, %scales)
            : (tensor<1x3x4x6xf32>, tensor<2xf32>)
                -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        return %result : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>

        // Call site forwards the f32 scales unchanged, matching the @foo signature.
        // CHECK-NOT:   IE.Convert({{.*}}) {dstElemType = f16} : tensor<2xf32>
        // CHECK:       [[RES:%.+]] = call @foo([[DATA]], [[SCALES]])
        // CHECK-SAME:    : (tensor<1x3x4x6xf16>, tensor<2xf32>)
        // CHECK-SAME:        -> tensor<1x3x?x?xf16
        // CHECK:       return [[RES]]
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @CalleeScalesAlsoUsedByMultiplyNotPreserved
// The forwarded scales feeds a SAP Interpolate AND a Multiply inside the callee.
// The non-Interpolate consumer breaks the forward cone, so the whole chain is
// converted to f16 (no f32 preservation) and stays type-consistent.
module @CalleeScalesAlsoUsedByMultiplyNotPreserved {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            DataInfo "data" : tensor<1x3x4x6xf32>
            DataInfo "scales" : tensor<2xf32>
        }
        outputsInfo : {
            DataInfo "interp" : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
            DataInfo "scaled" : tensor<2xf32>
        }

    // CHECK:       func.func @foo(
    // CHECK-SAME:      [[FOO_DATA:%[^:]+]]: tensor<1x3x4x6xf16>,
    // CHECK-SAME:      [[FOO_SCALES:%[^:]+]]: tensor<2xf16>
    func.func @foo(%data: tensor<1x3x4x6xf32>, %scales: tensor<2xf32>)
            -> (tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>,
                tensor<2xf32>) {
        %interp = IE.Interpolate(%data, %scales) {
            attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>,
                                   nearest_mode = <FLOOR>, antialias = false,
                                   pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
                                   cube_coeff = -7.500000e-01 : f64>,
            axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 1, 0>, sizes_attr = []}
            : tensor<1x3x4x6xf32>, tensor<2xf32>
                -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        %scaled = IE.Multiply(%scales, %scales) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
        return %interp, %scaled
            : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>,
              tensor<2xf32>

        // Multiply user blocks preservation: scales is narrowed to f16.
        // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[FOO_DATA]], [[FOO_SCALES]])
        // CHECK-SAME:    : tensor<1x3x4x6xf16>, tensor<2xf16>
        // CHECK:       [[SCALED:%.+]] = IE.Multiply([[FOO_SCALES]], [[FOO_SCALES]])
        // CHECK-SAME:    : tensor<2xf16>, tensor<2xf16> -> tensor<2xf16>
        // CHECK:       return [[INTERP]], [[SCALED]]
    }

    // CHECK:       func.func @main(
    // CHECK-SAME:      [[DATA:%[^:]+]]: tensor<1x3x4x6xf16>,
    // CHECK-SAME:      [[SCALES:%[^:]+]]: tensor<2xf16>
    func.func @main(%data: tensor<1x3x4x6xf32>, %scales: tensor<2xf32>)
            -> (tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>,
                tensor<2xf32>) {
        %interp, %scaled = func.call @foo(%data, %scales)
            : (tensor<1x3x4x6xf32>, tensor<2xf32>)
                -> (tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>,
                    tensor<2xf32>)
        return %interp, %scaled
            : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>,
              tensor<2xf32>

        // Call site is consistent: scales forwarded as f16.
        // CHECK:       call @foo([[DATA]], [[SCALES]])
        // CHECK-SAME:    : (tensor<1x3x4x6xf16>, tensor<2xf16>)
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ScalesSharedAcrossCallsNotPreserved
// Same idea as @ScalesSharedWithMultiplyNotPreserved, but the blocking
// non-Interpolate consumer lives behind a func.call.
// Preserving the Interpolate branch here would require materializing a separate
// f32 copy of the scales for the Interpolate-only call. TODO: E#221311
module @ScalesSharedAcrossCallsNotPreserved {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            DataInfo "data" : tensor<1x3x4x6xf32>
            DataInfo "scales" : tensor<2xf32>
        }
        outputsInfo : {
            DataInfo "interp" : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
            DataInfo "scaled" : tensor<2xf32>
        }

    // CHECK:       func.func @consumeScales(
    // CHECK-SAME:      [[CS_SCALES:%[^:]+]]: tensor<2xf16>
    func.func @consumeScales(%scales: tensor<2xf32>) -> tensor<2xf32> {
        %scaled = IE.Multiply(%scales, %scales) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
        return %scaled : tensor<2xf32>

        // CHECK:       [[SCALED:%.+]] = IE.Multiply([[CS_SCALES]], [[CS_SCALES]])
        // CHECK-SAME:    : tensor<2xf16>, tensor<2xf16> -> tensor<2xf16>
        // CHECK:       return [[SCALED]]
    }

    // CHECK:       func.func @interpScales(
    // CHECK-SAME:      [[IS_DATA:%[^:]+]]: tensor<1x3x4x6xf16>,
    // CHECK-SAME:      [[IS_SCALES:%[^:]+]]: tensor<2xf16>
    func.func @interpScales(%data: tensor<1x3x4x6xf32>, %scales: tensor<2xf32>)
            -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}> {
        %interp = IE.Interpolate(%data, %scales) {
            attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>,
                                   nearest_mode = <FLOOR>, antialias = false,
                                   pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
                                   cube_coeff = -7.500000e-01 : f64>,
            axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 1, 0>, sizes_attr = []}
            : tensor<1x3x4x6xf32>, tensor<2xf32>
                -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        return %interp : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>

        // The shared scales is not preserved, so the Interpolate scales is f16 here too.
        // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[IS_DATA]], [[IS_SCALES]])
        // CHECK-SAME:    : tensor<1x3x4x6xf16>, tensor<2xf16>
        // CHECK:       return [[INTERP]]
    }

    // CHECK:       func.func @main(
    // CHECK-SAME:      [[DATA:%[^:]+]]: tensor<1x3x4x6xf16>,
    // CHECK-SAME:      [[SCALES:%[^:]+]]: tensor<2xf16>
    func.func @main(%data: tensor<1x3x4x6xf32>, %scales: tensor<2xf32>)
            -> (tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>,
                tensor<2xf32>) {
        %scaled = func.call @consumeScales(%scales) : (tensor<2xf32>) -> tensor<2xf32>
        %interp = func.call @interpScales(%data, %scales)
            : (tensor<1x3x4x6xf32>, tensor<2xf32>)
                -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        return %interp, %scaled
            : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>,
              tensor<2xf32>

        // Both call sites forward the scales as f16; no f32 is preserved.
        // CHECK:       [[SCALED:%.+]] = call @consumeScales([[SCALES]]) : (tensor<2xf16>) -> tensor<2xf16>
        // CHECK:       [[INTERP:%.+]] = call @interpScales([[DATA]], [[SCALES]])
        // CHECK-SAME:    : (tensor<1x3x4x6xf16>, tensor<2xf16>)
        // CHECK:       return [[INTERP]], [[SCALED]]
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @ScalesSharedAcrossCallsNotPreserved_ReverseCallOrder
// Same as @ScalesSharedAcrossCallsNotPreserved, but the SAP-Interpolate call
// precedes the blocking consumer call.
module @ScalesSharedAcrossCallsNotPreserved_ReverseCallOrder {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            DataInfo "data" : tensor<1x3x4x6xf32>
            DataInfo "scales" : tensor<2xf32>
        }
        outputsInfo : {
            DataInfo "interp" : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
            DataInfo "scaled" : tensor<2xf32>
        }

    // CHECK:       func.func @consumeScales(
    // CHECK-SAME:      [[CS_SCALES:%[^:]+]]: tensor<2xf16>
    func.func @consumeScales(%scales: tensor<2xf32>) -> tensor<2xf32> {
        %scaled = IE.Multiply(%scales, %scales) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
        return %scaled : tensor<2xf32>

        // CHECK:       [[SCALED:%.+]] = IE.Multiply([[CS_SCALES]], [[CS_SCALES]])
        // CHECK-SAME:    : tensor<2xf16>, tensor<2xf16> -> tensor<2xf16>
        // CHECK:       return [[SCALED]]
    }

    // CHECK:       func.func @interpScales(
    // CHECK-SAME:      [[IS_DATA:%[^:]+]]: tensor<1x3x4x6xf16>,
    // CHECK-SAME:      [[IS_SCALES:%[^:]+]]: tensor<2xf16>
    func.func @interpScales(%data: tensor<1x3x4x6xf32>, %scales: tensor<2xf32>)
            -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}> {
        %interp = IE.Interpolate(%data, %scales) {
            attr = #IE.Interpolate<mode = <LINEAR>, shape_calc_mode = <SCALES>, coord_mode = <HALF_PIXEL>,
                                   nearest_mode = <FLOOR>, antialias = false,
                                   pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
                                   cube_coeff = -7.500000e-01 : f64>,
            axes_attr = [2, 3], operandSegmentSizes = array<i32: 1, 0, 1, 0>, sizes_attr = []}
            : tensor<1x3x4x6xf32>, tensor<2xf32>
                -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        return %interp : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>

        // The shared scales is not preserved, so the Interpolate scales is f16 here too.
        // CHECK:       [[INTERP:%.+]] = IE.Interpolate([[IS_DATA]], [[IS_SCALES]])
        // CHECK-SAME:    : tensor<1x3x4x6xf16>, tensor<2xf16>
        // CHECK:       return [[INTERP]]
    }

    // CHECK:       func.func @main(
    // CHECK-SAME:      [[DATA:%[^:]+]]: tensor<1x3x4x6xf16>,
    // CHECK-SAME:      [[SCALES:%[^:]+]]: tensor<2xf16>
    func.func @main(%data: tensor<1x3x4x6xf32>, %scales: tensor<2xf32>)
            -> (tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>,
                tensor<2xf32>) {
        %interp = func.call @interpScales(%data, %scales)
            : (tensor<1x3x4x6xf32>, tensor<2xf32>)
                -> tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>
        %scaled = func.call @consumeScales(%scales) : (tensor<2xf32>) -> tensor<2xf32>
        return %interp, %scaled
            : tensor<1x3x?x?xf32, {bounds = #const.OpaqueI64Elements<[1, 3, 32, 48]> : tensor<4xsi64>, order = #NCHW}>,
              tensor<2xf32>

        // Both call sites forward the scales as f16; no f32 is preserved.
        // CHECK:       [[INTERP:%.+]] = call @interpScales([[DATA]], [[SCALES]])
        // CHECK-SAME:    : (tensor<1x3x4x6xf16>, tensor<2xf16>)
        // CHECK:       [[SCALED:%.+]] = call @consumeScales([[SCALES]]) : (tensor<2xf16>) -> tensor<2xf16>
        // CHECK:       return [[INTERP]], [[SCALED]]
    }
}
