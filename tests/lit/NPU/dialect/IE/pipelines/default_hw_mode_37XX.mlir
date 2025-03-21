//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Convolution
module @Convolution {

    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
    func.func @main(%arg: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %1 = IE.Convolution(%arg, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        return %1 : tensor<1x48x60x60xf32>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> :
        // CHECK-SAME:       tensor<48x3x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]

        // CHECK:       [[EXPAND:%.+]] = IE.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        // CHECK:       [[PERM:%.+]] = IE.PermuteQuantize([[EXPAND]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} :
        // CHECK-SAME:       tensor<1x3x62x64xf16> -> tensor<1x16x62x64xf16, {order = #NHWC}>
        // CHECK:       [[SLICE:%.+]] = IE.Slice [[PERM]] [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>

        // CHECK:       [[OUT:%.+]] = IE.Convolution([[SLICE]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
        // CHECK-SAME:       tensor<1x16x62x62xf16, {order = #NHWC}>, tensor<48x16x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        // CHECK:       return [[OUT]] : tensor<1x48x60x60xf16>
    }
}

// -----

// CHECK-LABEL: @ReduceMax
module @ReduceMax {

IE.CNNNetwork
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x8x24x24xf16>
    }
    outputsInfo : {
        DataInfo "softmax" : tensor<1x1x24x24xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x8x24x24xf16>) -> tensor<1x1x24x24xf16> {
    func.func @main(%arg0: tensor<1x8x24x24xf16>) -> tensor<1x1x24x24xf16> {
        %0 = IE.ReduceMax(%arg0) {axes_value = [1], keep_dims} : tensor<1x8x24x24xf16> -> tensor<1x1x24x24xf16>
        return %0 : tensor<1x1x24x24xf16>

        // CHECK:               [[RESHAPE0:%.+]] = IE.Reshape({{[^:]+}}) {shape_value = [1, 8, 36, 16]} : tensor<1x8x24x24xf16> -> tensor<1x8x36x16xf16>
        // CHECK:               [[PERMUTECAST0:%.+]] = IE.PermuteCast([[RESHAPE0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x8x36x16xf16> -> tensor<1x16x8x36xf16, {order = #NHWC}>
        // CHECK:               [[MAXPOOL:%.+]] = IE.MaxPool([[PERMUTECAST0]]) {kernel_size = [8, 1], pads_begin = [0, 0], pads_end = [0, 0],
        // CHECK-SAME:            rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x8x36xf16, {order = #NHWC}> -> tensor<1x16x1x36xf16, {order = #NHWC}>
        // CHECK:               [[PERMUTECAST1:%.+]] = IE.PermuteCast([[MAXPOOL]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x16x1x36xf16, {order = #NHWC}> -> tensor<1x1x36x16xf16>
        // CHECK:               [[RESHAPE1:%.+]] = IE.Reshape([[PERMUTECAST1]]) {shape_value = [1, 1, 24, 24]} : tensor<1x1x36x16xf16> -> tensor<1x1x24x24xf16>
        // CHECK:               return [[RESHAPE1]] : tensor<1x1x24x24xf16>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TwoFunctions
module @TwoFunctions {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xui8>
    } outputsInfo : {
        DataInfo "output" : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @foo1([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
    func.func @foo1(%arg: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
        %cst = const.Declare tensor<48x3x3x3xf32> = dense<1.0> : tensor<48x3x3x3xf32>
        %0 = IE.Convolution(%arg, %cst) {
            dilations = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [1, 1]
        } : tensor<1x3x62x62xf32>, tensor<48x3x3x3xf32> -> tensor<1x48x60x60xf32>
        return %0 : tensor<1x48x60x60xf32>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<48x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> :
        // CHECK-SAME:       tensor<48x3x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]

        // CHECK:       [[EXPAND:%.+]] = IE.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        // CHECK:       [[PERM:%.+]] = IE.PermuteQuantize([[EXPAND]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 13, 0, 0]} :
        // CHECK-SAME:       tensor<1x3x62x64xf16> -> tensor<1x16x62x64xf16, {order = #NHWC}>
        // CHECK:       [[SLICE:%.+]] = IE.Slice [[PERM]] [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>

        // CHECK:       [[OUT:%.+]] = IE.Convolution([[SLICE]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
        // CHECK-SAME:       tensor<1x16x62x62xf16, {order = #NHWC}>, tensor<48x16x3x3xf16, {order = #NHWC}> -> tensor<1x48x60x60xf16>
        // CHECK:       return [[OUT]] : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @foo2([[ARG0:%.+]]: tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
    func.func @foo2(%arg: tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32> {
        %0 = IE.SoftMax(%arg) {axisInd = 3} : tensor<1x48x60x60xf32> -> tensor<1x48x60x60xf32>
        return %0 : tensor<1x48x60x60xf32>

        // CHECK: [[SOFTMAX:%.+]] = IE.SoftMax([[ARG0]]) {axisInd = 3 : i64} : tensor<1x48x60x60xf16> -> tensor<1x48x60x60xf16>
        // CHECK: return [[SOFTMAX]] : tensor<1x48x60x60xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x3x62x62xui8>) -> tensor<1x48x60x60xf16>
    func.func @main(%arg: tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32> {
        %0 = call @foo1(%arg) : (tensor<1x3x62x62xf32>) -> tensor<1x48x60x60xf32>
        %1 = call @foo2(%0) : (tensor<1x48x60x60xf32>) -> tensor<1x48x60x60xf32>
        return %1 : tensor<1x48x60x60xf32>

        // CHECK: [[CONVERT:%.+]] = IE.Convert([[ARG0]]) {dstElemType = f16} : tensor<1x3x62x62xui8> -> tensor<1x3x62x62xf16>
        // CHECK: [[FOO1_RES:%.+]] = call @foo1([[CONVERT]]) : (tensor<1x3x62x62xf16>) -> tensor<1x48x60x60xf16>
        // CHECK: [[FOO2_RES:%.+]] = call @foo2([[FOO1_RES]]) : (tensor<1x48x60x60xf16>) -> tensor<1x48x60x60xf16>
        // CHECK: return [[FOO2_RES]] : tensor<1x48x60x60xf16>
    }
}

// -----

// CHECK-LABEL-DAG: @MatMulWithGroupQuant
// CHECK-DAG: [[Q_TYPE:!.*]] = !quant.uniform<i4:f16, 2.000000e+00>
// CHECK-DAG: [[Q_TYPE1:!.*]] = !quant.uniform<u4:f16, 2.000000e+00:8>
module @MatMulWithGroupQuant {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<16x3072xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<16x4096xf32>
    }

    // CHECK: func.func @main
    // CHECK-SAME: [[ARG:%.+]]: tensor<16x3072xf32>
    func.func @main(%arg0: tensor<16x3072xf32>) -> tensor<16x4096xf32> {
        %WEIGHTS = const.Declare tensor<3x1024x4096xf32> = dense<1.0> : tensor<3x1024x4096xf32>
        // CHECK-DAG:   [[WEIGHTS_0:%.*]] = const.Declare tensor<4096x1024x1x1x[[Q_TYPE]], {order = #NHWC}> = dense<1.000000e+00> : tensor<3x1024x4096xf32>, [#const.SubView<[0, 0, 0], [1, 1024, 4096]>, #const.Reshape<[1, 1024, 1, 4096]>, #const.CastElemType<f16>, #const.CastElemType<[[Q_TYPE1]]>, #const.Transpose<#NWHC>, #const.Reshape<[4096, 1024, 1, 1]>, #const.ConvertElemType<[[Q_TYPE]]>, #const.Reorder<#NHWC>]
        // CHECK-DAG:   [[WEIGHTS_1:%.*]] = const.Declare tensor<4096x1024x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x1024x4096xf32>, [#const.SubView<[1, 0, 0], [1, 1024, 4096]>, #const.Reshape<[1, 1024, 1, 4096]>, #const.CastElemType<f16>, #const.CastElemType<[[Q_TYPE1]]>, #const.Transpose<#NWHC>, #const.Reshape<[4096, 1024, 1, 1]>, #const.ConvertElemType<[[Q_TYPE]]>, #const.Reorder<#NHWC>]
        // CHECK-DAG:   [[WEIGHTS_2:%.*]] = const.Declare tensor<4096x1024x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x1024x4096xf32>, [#const.SubView<[2, 0, 0], [1, 1024, 4096]>, #const.Reshape<[1, 1024, 1, 4096]>, #const.CastElemType<f16>, #const.CastElemType<[[Q_TYPE1]]>, #const.Transpose<#NWHC>, #const.Reshape<[4096, 1024, 1, 1]>, #const.ConvertElemType<[[Q_TYPE]]>, #const.Reorder<#NHWC>]

        // CHECK:   [[RESHAPE_LHS:%.*]] = IE.AffineReshape([[ARG]]) {
        // CHECK-SAME:      shape_value = [1, 1, 16, 3072]
        // CHECK-SAME:  } : tensor<16x3072xf32> -> tensor<1x1x16x3072xf32>
        // CHECK:   [[CONVERT_LHS:%.*]] = IE.Convert([[RESHAPE_LHS]]) {
        // CHECK-SAME:      dstElemType = f16
        // CHECK-SAME:  } : tensor<1x1x16x3072xf32> -> tensor<1x1x16x3072xf16>

        %IN_LOW = const.Declare tensor<1x1x1xf32> = dense<0.0e+00> : tensor<1x1x1xf32>
        %IN_HIGH = const.Declare tensor<1x1x1xf32> = dense<1.5e+01> : tensor<1x1x1xf32>
        %OUT_LOW = const.Declare tensor<3x1x4096xf32> = dense<-16.0> : tensor<3x1x4096xf32>
        %OUT_HIGH = const.Declare tensor<3x1x4096xf32> = dense<14.0> : tensor<3x1x4096xf32>

        %FQ = IE.FakeQuantize(%WEIGHTS, %IN_LOW, %IN_HIGH, %OUT_LOW, %OUT_HIGH) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            levels = 16 : i64
        } : tensor<3x1024x4096xf32>,
            tensor<1x1x1xf32>,
            tensor<1x1x1xf32>,
            tensor<3x1x4096xf32>,
            tensor<3x1x4096xf32>
                -> tensor<3x1024x4096xf32>

        %SHAPE_CST = const.Declare tensor<2xsi64> = dense<[3072, 4096]> : tensor<2xsi64>
        %RESHAPE = IE.Reshape(%FQ, %SHAPE_CST) : tensor<3x1024x4096xf32>, tensor<2xsi64> -> tensor<3072x4096xf32>
        %GEMM = IE.MatMul(%arg0, %RESHAPE) : tensor<16x3072xf32>, tensor<3072x4096xf32> -> tensor<16x4096xf32>
        // CHECK:   [[SLICE_0:%.*]] = IE.Slice [[CONVERT_LHS]] [0, 0, 0, 0] [1, 1, 16, 1024] : tensor<1x1x16x3072xf16> to tensor<1x1x16x1024xf16>
        // CHECK:   [[SLICE_1:%.*]] = IE.Slice [[CONVERT_LHS]] [0, 0, 0, 1024] [1, 1, 16, 1024] : tensor<1x1x16x3072xf16> to tensor<1x1x16x1024xf16>
        // CHECK:   [[SLICE_2:%.*]] = IE.Slice [[CONVERT_LHS]] [0, 0, 0, 2048] [1, 1, 16, 1024] : tensor<1x1x16x3072xf16> to tensor<1x1x16x1024xf16>

        // CHECK:   [[RESHAPE_SLICE_0:%.*]] = IE.AffineReshape([[SLICE_0]]) {
        // CHECK-SAME:      shape_value = [16, 1024, 1, 1]
        // CHECK-SAME:  } : tensor<1x1x16x1024xf16> -> tensor<16x1024x1x1xf16>

        // CHECK:   [[PERMUTE_CAST_SLICE_0:%.*]] = IE.PermuteCast([[RESHAPE_SLICE_0]]) {
        // CHECK-SAME:      dst_order = #NHWC,
        // CHECK-SAME:      mem_perm = #map
        // CHECK-SAME:  } : tensor<16x1024x1x1xf16> -> tensor<1x1024x16x1xf16, {order = #NHWC}>

        // CHECK:   [[CONV_INPUT_0:%.*]] = IE.AffineReshape([[PERMUTE_CAST_SLICE_0]]) {
        // CHECK-SAME:      shape_value = [1, 1024, 4, 4]
        // CHECK-SAME:  } : tensor<1x1024x16x1xf16, {order = #NHWC}> -> tensor<1x1024x4x4xf16, {order = #NHWC}>
        // CHECK:   [[CONV_0:%.*]] = IE.Convolution([[CONV_INPUT_0]], [[WEIGHTS_0]]) {
        // CHECK-SAME:      dilations = [1, 1],
        // CHECK-SAME:      pads_begin = [0, 0],
        // CHECK-SAME:      pads_end = [0, 0],
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME:  } : tensor<1x1024x4x4xf16, {order = #NHWC}>, tensor<4096x1024x1x1x!qElemType, {order = #NHWC}> -> tensor<1x4096x4x4xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_CONV_0:%.*]] = IE.AffineReshape([[CONV_0]]) {
        // CHECK-SAME:      shape_value = [1, 4096, 16, 1]
        // CHECK-SAME:  } : tensor<1x4096x4x4xf16, {order = #NHWC}> -> tensor<1x4096x16x1xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_SLICE_1:%.*]] = IE.AffineReshape([[SLICE_1]]) {
        // CHECK-SAME:      shape_value = [16, 1024, 1, 1]
        // CHECK-SAME:  } : tensor<1x1x16x1024xf16> -> tensor<16x1024x1x1xf16>

        // CHECK:   [[PERMUTE_CAST_SLICE_1:%.*]] = IE.PermuteCast([[RESHAPE_SLICE_1]]) {
        // CHECK-SAME:      dst_order = #NHWC,
        // CHECK-SAME:      mem_perm = #map
        // CHECK-SAME:  } : tensor<16x1024x1x1xf16> -> tensor<1x1024x16x1xf16, {order = #NHWC}>

        // CHECK:   [[CONV_INPUT_1:%.*]] = IE.AffineReshape([[PERMUTE_CAST_SLICE_1]]) {
        // CHECK-SAME:      shape_value = [1, 1024, 4, 4]
        // CHECK-SAME:  } : tensor<1x1024x16x1xf16, {order = #NHWC}> -> tensor<1x1024x4x4xf16, {order = #NHWC}>
        // CHECK:   [[CONV_1:%.*]] = IE.Convolution([[CONV_INPUT_1]], [[WEIGHTS_1]]) {
        // CHECK-SAME:      dilations = [1, 1],
        // CHECK-SAME:      pads_begin = [0, 0],
        // CHECK-SAME:      pads_end = [0, 0],
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME:  } : tensor<1x1024x4x4xf16, {order = #NHWC}>, tensor<4096x1024x1x1x!qElemType, {order = #NHWC}> -> tensor<1x4096x4x4xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_CONV_1:%.*]] = IE.AffineReshape([[CONV_1]]) {
        // CHECK-SAME:      shape_value = [1, 4096, 16, 1]
        // CHECK-SAME:  } : tensor<1x4096x4x4xf16, {order = #NHWC}> -> tensor<1x4096x16x1xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_SLICE_2:%.*]] = IE.AffineReshape([[SLICE_2]]) {
        // CHECK-SAME:      shape_value = [16, 1024, 1, 1]
        // CHECK-SAME:  } : tensor<1x1x16x1024xf16> -> tensor<16x1024x1x1xf16>

        // CHECK:   [[PERMUTE_CAST_SLICE_2:%.*]] = IE.PermuteCast([[RESHAPE_SLICE_2]]) {
        // CHECK-SAME:      dst_order = #NHWC,
        // CHECK-SAME:      mem_perm = #map
        // CHECK-SAME:  } : tensor<16x1024x1x1xf16> -> tensor<1x1024x16x1xf16, {order = #NHWC}>

        // CHECK:   [[CONV_INPUT_2:%.*]] = IE.AffineReshape([[PERMUTE_CAST_SLICE_2]]) {
        // CHECK-SAME:      shape_value = [1, 1024, 4, 4]
        // CHECK-SAME:  } : tensor<1x1024x16x1xf16, {order = #NHWC}> -> tensor<1x1024x4x4xf16, {order = #NHWC}>
        // CHECK:   [[CONV_2:%.*]] = IE.Convolution([[CONV_INPUT_2]], [[WEIGHTS_2]]) {
        // CHECK-SAME:      dilations = [1, 1],
        // CHECK-SAME:      pads_begin = [0, 0],
        // CHECK-SAME:      pads_end = [0, 0],
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME:  } : tensor<1x1024x4x4xf16, {order = #NHWC}>, tensor<4096x1024x1x1x!qElemType, {order = #NHWC}> -> tensor<1x4096x4x4xf16, {order = #NHWC}>

        // CHECK:   [[RESHAPE_CONV_2:%.*]] = IE.AffineReshape([[CONV_2]]) {
        // CHECK-SAME:      shape_value = [1, 4096, 16, 1]
        // CHECK-SAME:  } : tensor<1x4096x4x4xf16, {order = #NHWC}> -> tensor<1x4096x16x1xf16, {order = #NHWC}>

        // CHECK:   [[ADD_0:%.*]] = IE.Accumulate([[RESHAPE_CONV_0]], [[RESHAPE_CONV_1]])
        // CHECK:   [[ADD_1:%.*]] = IE.Accumulate([[ADD_0]], [[RESHAPE_CONV_2]])

        // CHECK:   [[PERMUTE_CAST_OUT:%.*]] = IE.PermuteCast([[ADD_1]]) {
        // CHECK-SAME:      dst_order = #NCHW,
        // CHECK-SAME:      mem_perm = #NHCW
        // CHECK-SAME:  } : tensor<1x4096x16x1xf16, {order = #NHWC}> -> tensor<1x1x16x4096xf16>

        // CHECK:   [[CONVERT_OUT:%.*]] = IE.Convert([[PERMUTE_CAST_OUT]]) {
        // CHECK-SAME:      dstElemType = f32
        // CHECK-SAME:  } : tensor<1x1x16x4096xf16> -> tensor<1x1x16x4096xf32>

        // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[CONVERT_OUT]]) {
        // CHECK-SAME:      shape_value = [16, 4096]
        // CHECK-SAME:  } : tensor<1x1x16x4096xf32> -> tensor<16x4096xf32>

        return %GEMM : tensor<16x4096xf32>
        // CHECK:   return [[RESHAPE_OUT]] : tensor<16x4096xf32>
    }
}

// -----

// CHECK-LABEL: @RMSProcessingWith2DRMS
module @RMSProcessingWith2DRMS {
    IE.CNNNetwork entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x768xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x768xf32>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x768xf32>) -> tensor<1x768xf32> {
    func.func @main(%arg0: tensor<1x768xf32>) -> tensor<1x768xf32> {

        %weight = const.Declare tensor<1x768xf32> = dense<1.0> : tensor<1x768xf32>
        %cst = IE.Reshape(%weight) {shape_value = [768]} : tensor<1x768xf32> -> tensor<768xf32>
        %out = IE.RMS(%arg0, %cst) {epsilon = 1.0013580322265625E-5 : f64} : tensor<1x768xf32>, tensor<768xf32> -> tensor<1x768xf32>

        return %out : tensor<1x768xf32>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<1x1x1x768xf16> = dense<1.000000e+00> : tensor<1x768xf32>, [#const.Reshape<[1, 1, 1, 768]>, #const.CastElemType<f16>]
        // CHECK:       [[AFFINE_RESHAPE_0:%.+]] = IE.AffineReshape(%arg0)
        // CHECK-LITERAL:   {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 768]} : tensor<1x768xf32> -> tensor<1x1x1x768xf32>
        // CHECK:       [[CONVERT_0:%.+]] = IE.Convert([[AFFINE_RESHAPE_0]]) {dstElemType = f16} : tensor<1x1x1x768xf32> -> tensor<1x1x1x768xf16>
        // CHECK:       [[RMS:%.+]] = IE.RMS([[CONVERT_0]], [[CST]]) {epsilon = 1.0013580322265625E-5 : f64} : tensor<1x1x1x768xf16>, tensor<1x1x1x768xf16> -> tensor<1x1x1x768xf16>
        // CHECK:       [[CONVERT_1:%.+]] = IE.Convert([[RMS]]) {dstElemType = f32} : tensor<1x1x1x768xf16> -> tensor<1x1x1x768xf32>
        // CHECK:       [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[CONVERT_1]])
        // CHECK-LITERAL:   {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 768]} : tensor<1x1x1x768xf32> -> tensor<1x768xf32>
        // CHECK:       return [[AFFINE_RESHAPE_1]] : tensor<1x768xf32>
}
}
