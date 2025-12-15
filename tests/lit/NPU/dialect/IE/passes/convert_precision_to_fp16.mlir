//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-precision-to-fp16 --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

//
// The 'convert-precision-to-fp16' pass:
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
    // CHECK:       %[[OUT:.+]] = IE.SoftMax([[ARG0]])
    // CHECK-SAME:      tensor<1x1000xf16> -> tensor<1x1000xf16>

    return %prob : tensor<1x1000xf32>
    // CHECK: return %[[OUT]] : tensor<1x1000xf16>
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

    // CHECK-DAG:       %[[OUT:.+]] = const.Declare tensor<1x2x2x2xf16> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<1x2x2x2xf32>, [#const.CastElemType<f16>]
    // CHECK:       return %[[OUT]] : tensor<1x2x2x2xf16>
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

        // CHECK-DAG: %[[NEG:.+]] = const.Declare {{.*}} = dense<-6.550400e+04> : tensor<1x2x2x2xf16>, [#const.CastElemType<f16>]
        // CHECK-DAG: %[[POS:.+]] = const.Declare {{.*}} = dense<6.550400e+04> : tensor<1x2x2x2xf16>, [#const.CastElemType<f16>]
        // CHECK: return %[[NEG]], %[[POS]]
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

        // CHECK: %[[ARR:.+]] = const.Declare
        // CHECK-SAME{LITERAL}: dense<[[[[-1.000000e+05, 1.000000e+05]]]]> : tensor<1x1x1x2xf32>, [#const.CastElemType<f16>]
        // CHECK: return %[[ARR]]
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

    // CHECK:  %0 = IE.And([[ARG0]], [[ARG1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1xf16>, tensor<1xf16> -> tensor<1xf16>
    // CHECK:  return %0 : tensor<1xf16>
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
    %0 = IE.OneHot(%arg0) {axis_attr = 0 : i64, depth_attr = 3 : i64, off_value_attr = 0.000000e+00 : f64, on_value_attr = 1.000000e+00 : f64, operandSegmentSizes = array<i32: 1, 0, 0, 0>, outputType = f32} : tensor<4xsi32> -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>

    // CHECK:       %[[OUT:.+]] = IE.OneHot([[ARG0]])
    // CHECK-SAME:      outputType = f16
    // CHECK-SAME:      tensor<4xsi32> -> tensor<3x4xf16>
    // CHECK: return %[[OUT]] : tensor<3x4xf16>
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

    // CHECK:       %[[OUT:.+]] = IE.Eye([[ARG0]])
    // CHECK-SAME:      outputType = f16
    // CHECK-SAME:      tensor<1xsi32> -> tensor<128x128xf16>
    // CHECK: return %[[OUT]] : tensor<128x128xf16>
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

    // CHECK:       %[[OUT:.+]] = IE.Eye([[ARG0]])
    // CHECK-SAME:      outputType = f16
    // CHECK-SAME:      tensor<1xsi32> -> tensor<128x128xf16>
    // CHECK: return %[[OUT]] : tensor<128x128xf16>
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
    %5 = IE.Convert(%4) {dstElemType = f16} : tensor<1x1024xi8> -> tensor<1x1024xf16>
    return %5 : tensor<1x1024xf16>
    // CHECK:       [[CONVER:%.+]] = IE.Convert(%arg0) {dstElemType = i8} : tensor<1x1024xf16> -> tensor<1x1024xi8>

    // CHECK:       [[BITWISENOT:%.+]] = IE.BitwiseNot([[CONVER]]) : tensor<1x1024xi8> -> tensor<1x1024xi8>
    // CHECK:       [[BITWISEAND:%.+]] = IE.BitwiseAnd([[CONVER]], [[BITWISENOT]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1x1024xi8> -> tensor<1x1024xi8>
    // CHECK:       [[BITWISEOR:%.+]] = IE.BitwiseOr([[CONVER]], [[BITWISEAND]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1x1024xi8> -> tensor<1x1024xi8>
    // CHECK:       [[BITWISEXOR:%.+]] = IE.BitwiseXor([[CONVER]], [[BITWISEOR]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1x1024xi8> -> tensor<1x1024xi8>

    // CHECK:       [[CONVER1:%.+]] = IE.Convert([[BITWISEXOR]]) {dstElemType = f16} : tensor<1x1024xi8> -> tensor<1x1024xf16>
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

// CHECK-LABEL: @SelectOpSi32ToFP16
module @SelectOpSi32ToFP16 {

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
    %cst = const.Declare tensor<1xsi32> = dense<256> : tensor<1xsi32> isSplat
    %0 = IE.Select(%arg0, %cst, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1xsi32>, tensor<1x1024xsi32> -> tensor<1x1024xsi32>
    return %0 : tensor<1x1024xsi32>

    // CHECK-DAG: [[FLAG_CONST:%.+]] = const.Declare tensor<1xf16> = dense<256> : tensor<1xsi32>, [#const.CastElemType<f16>]
    // CHECK:     [[CONVERT_DATA:%.+]] = IE.Convert([[ARG1]])
    // CHECK-SAME:      {dstElemType = f16} : tensor<1x1024xsi32> -> tensor<1x1024xf16>
    // CHECK:     [[SELECT:%.+]] = IE.Select([[ARG0]], [[FLAG_CONST]], [[CONVERT_DATA]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x1024xf16>, tensor<1xf16>, tensor<1x1024xf16> -> tensor<1x1024xf16>
    // CHECK:     [[CONVERT_OUT:%.+]] = IE.Convert([[SELECT]])
    // CHECK-SAME:      {dstElemType = si32} : tensor<1x1024xf16> -> tensor<1x1024xsi32>
    // CHECK:     return [[CONVERT_OUT]] : tensor<1x1024xsi32>
}

}

// -----

// CHECK-LABEL: @SelectOpSi64ToFP16
module @SelectOpSi64ToFP16 {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        // CHECK: DataInfo "flag" : tensor<1x1024xi8>
        // CHECK: DataInfo "data" : tensor<1x1024xsi64>
        DataInfo "flag" : tensor<1x1024xi8>
        DataInfo "data" : tensor<1x1024xsi64>
    }
    outputsInfo : {
        // CHECK: DataInfo "out" : tensor<1x1024xsi64>
        DataInfo "out" : tensor<1x1024xsi64>
    }

// CHECK: @main([[ARG0:[^:]+]]: tensor<1x1024xf16>, [[ARG1:[^:]+]]: tensor<1x1024xsi64>)
// CHECK:  -> tensor<1x1024xsi64>
func.func @main(%arg0: tensor<1x1024xi8>, %arg1: tensor<1x1024xsi64>) -> tensor<1x1024xsi64> {
    %cst = const.Declare tensor<1xsi64> = dense<256> : tensor<1xsi64> isSplat
    %0 = IE.Select(%arg0, %cst, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1024xi8>, tensor<1xsi64>, tensor<1x1024xsi64> -> tensor<1x1024xsi64>
    return %0 : tensor<1x1024xsi64>

    // CHECK-DAG: [[FLAG_CONST:%.+]] = const.Declare tensor<1xf16> = dense<256> : tensor<1xsi64>, [#const.CastElemType<f16>]
    // CHECK:     [[CONVERT_DATA:%.+]] = IE.Convert([[ARG1]])
    // CHECK-SAME:      {dstElemType = f16} : tensor<1x1024xsi64> -> tensor<1x1024xf16>
    // CHECK:     [[SELECT:%.+]] = IE.Select([[ARG0]], [[FLAG_CONST]], [[CONVERT_DATA]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-SAME:      tensor<1x1024xf16>, tensor<1xf16>, tensor<1x1024xf16> -> tensor<1x1024xf16>
    // CHECK:     [[CONVERT_OUT:%.+]] = IE.Convert([[SELECT]])
    // CHECK-SAME:      {dstElemType = si64} : tensor<1x1024xf16> -> tensor<1x1024xsi64>
    // CHECK:     return [[CONVERT_OUT]] : tensor<1x1024xsi64>
}

}

// -----

// CHECK-LABEL: @I64Clamp
module @I64Clamp {
    net.NetworkInfo
        entryPoint : @main
        inputsInfo : {
            // CHECK: DataInfo "Input" : tensor<1x1x512x512xsi64>
            DataInfo "Input" : tensor<1x1x512x512xsi64>
        }
        outputsInfo : {
            // CHECK: DataInfo "Out" : tensor<1x1x512x512xsi64>
            DataInfo "Out" : tensor<1x1x512x512xsi64>
        }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x1x512x512xsi64>) -> tensor<1x1x512x512xsi64>
    func.func @main(%input: tensor<1x1x512x512xsi64>) -> tensor<1x1x512x512xsi64> {
        %0 = IE.Clamp(%input) {max = 5.110000e+02 : f64, min = 0.000000e+00 : f64} : tensor<1x1x512x512xsi64> -> tensor<1x1x512x512xsi64>
        return %0 : tensor<1x1x512x512xsi64>

        // CHECK: [[CONVERT_IN:%.+]] = IE.Convert([[INPUT]]) {dstElemType = f16} : tensor<1x1x512x512xsi64> -> tensor<1x1x512x512xf16>
        // CHECK: [[CLAMP:%.+]] = IE.Clamp([[CONVERT_IN]]) {max = 5.110000e+02 : f64, min = 0.000000e+00 : f64} : tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf16>
        // CHECK: [[CONVERT_OUT:%.+]] = IE.Convert([[CLAMP]]) {dstElemType = si64} : tensor<1x1x512x512xf16> -> tensor<1x1x512x512xsi64>
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
        %cst = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
        %0 = IE.ReduceL2(%arg0, %cst) {keep_dims} : tensor<1x192xf16>, tensor<1xsi64> -> tensor<1x1xf16>
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
        %cst = const.Declare tensor<1xsi64> = dense<1> : tensor<1xsi64>
        %0 = IE.ReduceL2(%arg0, %cst) {keep_dims} : tensor<1x192xf16>, tensor<1xsi64> -> tensor<1x1xf16>
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
