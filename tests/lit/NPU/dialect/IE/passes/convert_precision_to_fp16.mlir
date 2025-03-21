//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-precision-to-fp16 --canonicalize %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
//
// The 'convert-precision-to-fp16' pass:
//
//   * Updates both Function bodies and Function prototypes.
//   * It shouldn't touch user types defined in `IE.CNNNetwork`.
//   * It should update types for `Constant` operation.
//

// CHECK-LABEL: @FP32toFP16
module @FP32toFP16 {

IE.CNNNetwork
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
    // CHECK:       %[[OUT:.*]] = IE.SoftMax([[ARG0]])
    // CHECK-SAME:      tensor<1x1000xf16> -> tensor<1x1000xf16>

    return %prob : tensor<1x1000xf32>
    // CHECK: return %[[OUT]] : tensor<1x1000xf16>
}

}

// -----

// CHECK-LABEL: @ConstantLayer
module @ConstantLayer {

IE.CNNNetwork
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

    // CHECK-DAG:       %[[OUT:.*]] = const.Declare tensor<1x2x2x2xf16> =
    // CHECK-SAME:      dense<1.000000e+00> : tensor<1x2x2x2xf32>, [#const.CastElemType<f16>]
    // CHECK:       return %[[OUT]] : tensor<1x2x2x2xf16>
}

}

// -----

// CHECK-LABEL: @I8ToFp16
module @I8ToFp16 {

IE.CNNNetwork
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

IE.CNNNetwork
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

    // CHECK:       %[[OUT:.*]] = IE.OneHot([[ARG0]])
    // CHECK-SAME:      tensor<4xsi32> -> tensor<3x4xf16>
    // CHECK: return %[[OUT]] : tensor<3x4xf16>
}

}

// -----

// CHECK-LABEL: @FP32Eye
module @FP32Eye {

IE.CNNNetwork
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

    // CHECK: %[[OUT:.*]] = IE.Eye([[ARG0]])
    // CHECK-SAME: tensor<1xsi32> -> tensor<128x128xf16>
    // CHECK: return %[[OUT]] : tensor<128x128xf16>
}

}

// -----

// CHECK-LABEL: @FP16Eye
module @FP16Eye {

IE.CNNNetwork
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

    // CHECK: %[[OUT:.*]] = IE.Eye([[ARG0]])
    // CHECK-SAME: tensor<1xsi32> -> tensor<128x128xf16>
    // CHECK: return %[[OUT]] : tensor<128x128xf16>
}

}

// -----

// CHECK-LABEL: @TwoFunctions
module @TwoFunctions {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
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

IE.CNNNetwork
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
    IE.CNNNetwork
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
    IE.CNNNetwork
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

IE.CNNNetwork
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

IE.CNNNetwork
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
    IE.CNNNetwork
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
