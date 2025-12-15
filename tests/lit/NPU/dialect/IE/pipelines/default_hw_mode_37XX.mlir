//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Convolution
module @Convolution {

    net.NetworkInfo entryPoint : @main
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

net.NetworkInfo
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
    net.NetworkInfo entryPoint : @main
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

#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>

// CHECK-LABEL-DAG: @MatMulWithGroupQuant
// CHECK-DAG: [[Q_TYPE:!.*]] = !quant.uniform<i4:f16, 2.000000e+00>
// CHECK-DAG: [[Q_TYPE1:!.*]] = !quant.uniform<u4:f16, 2.000000e+00:8>
module @MatMulWithGroupQuant {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<16x3072xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<16x4096xf32>
    }

    // CHECK: func.func @main
    // CHECK-SAME: [[ARG:%.+]]: tensor<16x3072xf32>
    func.func @main(%arg0: tensor<16x3072xf32>) -> tensor<16x4096xf32> {
        %WEIGHTS = const.Declare tensor<3x1024x4096xf32> = dense<1.0> : tensor<3x1024x4096xf32>
        // CHECK-DAG:   [[WEIGHTS_0:%.*]] = const.Declare tensor<4096x1024x1x1x[[Q_TYPE]], {order = #NHWC}> = dense<1.000000e+00> : tensor<3x1024x4096xf32>, [#const.SubView<[0, 0, 0], [1, 1024, 4096]>, #const.Reshape<[1, 1024, 1, 4096]>, #const.CastElemType<f16>, #const.CastElemType<[[Q_TYPE1]]>, #const.Transpose<#NWHC>, #const.AffineReshape<{{\[\[}}0], [0], [0], [1, 2, 3]], [4096, 1024, 1, 1]>, #const.ConvertElemType<[[Q_TYPE]]>, #const.Reorder<#NHWC>]
        // CHECK-DAG:   [[WEIGHTS_1:%.*]] = const.Declare tensor<4096x1024x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x1024x4096xf32>, [#const.SubView<[1, 0, 0], [1, 1024, 4096]>, #const.Reshape<[1, 1024, 1, 4096]>, #const.CastElemType<f16>, #const.CastElemType<[[Q_TYPE1]]>, #const.Transpose<#NWHC>, #const.AffineReshape<{{\[\[}}0], [0], [0], [1, 2, 3]], [4096, 1024, 1, 1]>, #const.ConvertElemType<[[Q_TYPE]]>, #const.Reorder<#NHWC>]
        // CHECK-DAG:   [[WEIGHTS_2:%.*]] = const.Declare tensor<4096x1024x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x1024x4096xf32>, [#const.SubView<[2, 0, 0], [1, 1024, 4096]>, #const.Reshape<[1, 1024, 1, 4096]>, #const.CastElemType<f16>, #const.CastElemType<[[Q_TYPE1]]>, #const.Transpose<#NWHC>, #const.AffineReshape<{{\[\[}}0], [0], [0], [1, 2, 3]], [4096, 1024, 1, 1]>, #const.ConvertElemType<[[Q_TYPE]]>, #const.Reorder<#NHWC>]

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

        // CHECK:   [[ADD_0:%.+]] = IE.Add([[CONV_0]], [[CONV_1]])
        // CHECK:   [[ADD_1:%.+]] = IE.Add([[ADD_0]], [[CONV_2]])

        // CHECK:   [[RESHAPE_ADD_OUT:%.+]] = IE.AffineReshape([[ADD_1]]) {
        // CHECK-SAME:      shape_value = [1, 4096, 16, 1]
        // CHECK-SAME:  } : tensor<1x4096x4x4xf32, {order = #NHWC}> -> tensor<1x4096x16x1xf32, {order = #NHWC}>

        // CHECK:   [[PERMUTE_CAST_OUT:%.+]] = IE.PermuteCast([[RESHAPE_ADD_OUT]]) {
        // CHECK-SAME:      dst_order = #NCHW,
        // CHECK-SAME:      mem_perm = #map1
        // CHECK-SAME:  } : tensor<1x4096x16x1xf32, {order = #NHWC}> -> tensor<16x4096x1x1xf32>

        // CHECK:   [[RESHAPE_OUT_0:%.+]] = IE.AffineReshape([[PERMUTE_CAST_OUT]]) {
        // CHECK-SAME:      shape_value = [1, 1, 16, 4096]
        // CHECK-SAME:  } : tensor<16x4096x1x1xf32> -> tensor<1x1x16x4096xf32>

        // CHECK:   [[RESHAPE_OUT_1:%.+]] = IE.AffineReshape([[RESHAPE_OUT_0]]) {
        // CHECK-SAME:      shape_value = [16, 4096]
        // CHECK-SAME:  } : tensor<1x1x16x4096xf32> -> tensor<16x4096xf32>

        return %GEMM : tensor<16x4096xf32>
        // CHECK:   return [[RESHAPE_OUT_1]] : tensor<16x4096xf32>
    }
}

// -----

// CHECK-LABEL: @RMSProcessingWith2DRMS
module @RMSProcessingWith2DRMS {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x768xf32>
    } outputsInfo : {
        DataInfo "output" : tensor<1x768xf32>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x768xf32>) -> tensor<1x768xf32> {
    func.func @main(%arg0: tensor<1x768xf32>) -> tensor<1x768xf32> {

        %weight = const.Declare tensor<1x768xf32> = dense<1.0> : tensor<1x768xf32>
        %cst = IE.Reshape(%weight) {shape_value = [768]} : tensor<1x768xf32> -> tensor<768xf32>
        %out = IE.RMS(%arg0, %cst) {eps = 1.0013580322265625E-5 : f64} : tensor<1x768xf32>, tensor<768xf32> -> tensor<1x768xf32>

        return %out : tensor<1x768xf32>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<1x1x1x768xf16> = dense<1.000000e+00> : tensor<1x768xf32>, [#const.Reshape<[1, 1, 1, 768]>, #const.CastElemType<f16>]
        // CHECK:       [[AFFINE_RESHAPE_0:%.+]] = IE.AffineReshape(%arg0)
        // CHECK{LITERAL}:   {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 768]} : tensor<1x768xf32> -> tensor<1x1x1x768xf32>
        // CHECK:       [[CONVERT_0:%.+]] = IE.Convert([[AFFINE_RESHAPE_0]]) {dstElemType = f16} : tensor<1x1x1x768xf32> -> tensor<1x1x1x768xf16>
        // CHECK:       [[RMS:%.+]] = IE.RMS([[CONVERT_0]], [[CST]]) {eps = 1.0013580322265625E-5 : f64} : tensor<1x1x1x768xf16>, tensor<1x1x1x768xf16> -> tensor<1x1x1x768xf16>
        // CHECK:       [[CONVERT_1:%.+]] = IE.Convert([[RMS]]) {dstElemType = f32} : tensor<1x1x1x768xf16> -> tensor<1x1x1x768xf32>
        // CHECK:       [[AFFINE_RESHAPE_1:%.+]] = IE.AffineReshape([[CONVERT_1]])
        // CHECK{LITERAL}:   {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 768]} : tensor<1x1x1x768xf32> -> tensor<1x768xf32>
        // CHECK:       return [[AFFINE_RESHAPE_1]] : tensor<1x768xf32>
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Test the dependency relationship between ConvertGroupConvToConv and HandleLargeKernels
// It can convert GroupConv with large kernel to NCEConvolution
// CHECK-LABEL: @HandleGroupConvWithLargeKernels
module @HandleGroupConvWithLargeKernels {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x128x1x112xf16>
        DataInfo "input1" : tensor<128x64x1x22xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x128x1x91xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x128x1x112xf16>, [[ARG1:%.+]]: tensor<128x64x1x22xf16>) -> tensor<1x128x1x91xf16> {
    func.func @main(%arg0: tensor<1x128x1x112xf16>, %arg1: tensor<128x64x1x22xf16>) -> tensor<1x128x1x91xf16> {
        %group_conv = IE.GroupConvolution(%arg0, %arg1) {
                        dilations = [1, 1],
                        groups = 2,
                        pads_begin = [0, 0],
                        pads_end = [0, 0],
                        strides = [1, 1]
                    } : tensor<1x128x1x112xf16>, tensor<128x64x1x22xf16> -> tensor<1x128x1x91xf16>

        return %group_conv : tensor<1x128x1x91xf16>

        // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast([[ARG1]]) {dst_order = #NHWC, mem_perm = #map} : tensor<128x64x1x22xf16> -> tensor<1x22x128x64xf16, {order = #NHWC}>
        // CHECK:   [[SHAPECAST_0:%.+]] = IE.ShapeCast {shape = [1, 16, 128, 88]} inputs([[PERMUTECAST_0]] : tensor<1x22x128x64xf16, {order = #NHWC}>) -> tensor<1x16x128x88xf16, {order = #NHWC}>
        // CHECK:   [[MAXPOOL_0:%.+]] = IE.MaxPool([[SHAPECAST_0]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x128x88xf16, {order = #NHWC}> -> tensor<1x16x128x88xf16, {order = #NWCH}>
        // CHECK:   [[LAYOUTCAST_0:%.+]] = IE.LayoutCast([[MAXPOOL_0]]) {dst_order = #NHWC} : tensor<1x16x128x88xf16, {order = #NWCH}> -> tensor<1x16x128x88xf16, {order = #NHWC}>
        // CHECK:   [[SHAPECAST_1:%.+]] = IE.ShapeCast {shape = [1, 128, 64, 22]} inputs([[LAYOUTCAST_0]] : tensor<1x16x128x88xf16, {order = #NHWC}>) -> tensor<1x128x64x22xf16, {order = #NHWC}>
        // CHECK:   [[MAXPOOL_1:%.+]] = IE.MaxPool([[SHAPECAST_1]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x128x64x22xf16, {order = #NHWC}> -> tensor<1x128x64x22xf16, {order = #NCWH}>
        // CHECK:   [[PERMUTE_WEIGHT:%.+]] = IE.PermuteCast([[MAXPOOL_1]]) {dst_order = #NHWC, mem_perm = #map1} : tensor<1x128x64x22xf16, {order = #NCWH}> -> tensor<128x64x1x22xf16, {order = #NHWC}>

        // CHECK-DAG:   [[SLICE_WEIGHT_0:%.+]] = IE.Slice [[PERMUTE_WEIGHT]] [0, 0, 0, 0] [64, 64, 1, 11] : tensor<128x64x1x22xf16, {order = #NHWC}> to tensor<64x64x1x11xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SLICE_WEIGHT_1:%.+]] = IE.Slice [[PERMUTE_WEIGHT]] [0, 0, 0, 11] [64, 64, 1, 11] : tensor<128x64x1x22xf16, {order = #NHWC}> to tensor<64x64x1x11xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SLICE_WEIGHT_2:%.+]] = IE.Slice [[PERMUTE_WEIGHT]] [64, 0, 0, 0] [64, 64, 1, 11] : tensor<128x64x1x22xf16, {order = #NHWC}> to tensor<64x64x1x11xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SLICE_WEIGHT_3:%.+]] = IE.Slice [[PERMUTE_WEIGHT]] [64, 0, 0, 11] [64, 64, 1, 11] : tensor<128x64x1x22xf16, {order = #NHWC}> to tensor<64x64x1x11xf16, {order = #NHWC}>
        // CHECK:   [[PERMUTE_IN:%.+]] = IE.PermuteQuantize([[ARG0]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x128x1x112xf16> -> tensor<1x128x1x112xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_IN_3:%.+]] = IE.Slice [[PERMUTE_IN]] [0, 64, 0, 11] [1, 64, 1, 101] : tensor<1x128x1x112xf16, {order = #NHWC}> to tensor<1x64x1x101xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_IN_2:%.+]] = IE.Slice [[PERMUTE_IN]] [0, 64, 0, 0] [1, 64, 1, 101] : tensor<1x128x1x112xf16, {order = #NHWC}> to tensor<1x64x1x101xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_IN_1:%.+]] = IE.Slice [[PERMUTE_IN]] [0, 0, 0, 11] [1, 64, 1, 101] : tensor<1x128x1x112xf16, {order = #NHWC}> to tensor<1x64x1x101xf16, {order = #NHWC}>
        // CHECK:   [[SLICE_IN_0:%.+]] = IE.Slice [[PERMUTE_IN]] [0, 0, 0, 0] [1, 64, 1, 101] : tensor<1x128x1x112xf16, {order = #NHWC}> to tensor<1x64x1x101xf16, {order = #NHWC}>

        // CHECK:   [[CONV_0:%.+]] = IE.Convolution([[SLICE_IN_0]], [[SLICE_WEIGHT_0]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x101xf16, {order = #NHWC}>, tensor<64x64x1x11xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16, {order = #NHWC}>
        // CHECK:   [[CONV_1:%.+]] = IE.Convolution([[SLICE_IN_1]], [[SLICE_WEIGHT_1]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x101xf16, {order = #NHWC}>, tensor<64x64x1x11xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16, {order = #NHWC}>

        // CHECK:   [[GROUP_0:%.+]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x91xf16, {order = #NHWC}>, tensor<1x64x1x91xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16>


        // CHECK:   [[CONV_2:%.+]] = IE.Convolution([[SLICE_IN_2]], [[SLICE_WEIGHT_2]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x101xf16, {order = #NHWC}>, tensor<64x64x1x11xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16, {order = #NHWC}>
        // CHECK:   [[CONV_3:%.+]] = IE.Convolution([[SLICE_IN_3]], [[SLICE_WEIGHT_3]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x101xf16, {order = #NHWC}>, tensor<64x64x1x11xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16, {order = #NHWC}>

        // CHECK:   [[GROUP_1:%.+]] = IE.Add([[CONV_2]], [[CONV_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x91xf16, {order = #NHWC}>, tensor<1x64x1x91xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16>


        // CHECK:   [[CONCAT:%.+]] = IE.Concat([[GROUP_0]], [[GROUP_1]]) {
        // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x1x91xf16>, tensor<1x64x1x91xf16> -> tensor<1x128x1x91xf16>
        // CHECK:   return [[CONCAT]] : tensor<1x128x1x91xf16>
    }
}

// -----

// CHECK-LABEL: @ReduceMax
module @ReduceMax {

net.NetworkInfo
    entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x42840x17xf16>
    }
    outputsInfo : {
        DataInfo "reducemax" : tensor<1x42840x1xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<1x42840x17xf16>)
    func.func @main(%arg0: tensor<1x42840x17xf16>) -> tensor<1x42840x1xf16> {
         %cst = const.Declare tensor<1xsi64> = dense<2> : tensor<1xsi64>
        %0 = IE.ReduceMax(%arg0, %cst) {keep_dims} : tensor<1x42840x17xf16>, tensor<1xsi64> -> tensor<1x42840x1xf16>
        return %0 : tensor<1x42840x1xf16>
    }

        // CHECK:       [[SLICE1:%.+]] = IE.Slice [[ARG0]] [0, 0, 0] [1, 42840, 1] : tensor<1x42840x17xf16> to tensor<1x42840x1xf16>
        // CHECK:       [[CONCAT_1:%.+]] = IE.Concat([[ARG0]], [[SLICE1]]) {static_offsets = {{\[\[}}0, 0, 0], [0, 0, 17]]} : tensor<1x42840x17xf16>, tensor<1x42840x1xf16> -> tensor<1x42840x18xf16>
        // CHECK:       [[EXPAND:%.+]] = IE.Expand([[CONCAT_1]]) {pads_begin = [0, 0, 0], pads_end = [0, 0, 14]} : tensor<1x42840x18xf16> -> tensor<1x42840x32xf16>
        // CHECK:       [[AFFINERESHAPE1:%.+]] = IE.AffineReshape([[EXPAND]])
        // CHECK-SAME{LITERAL}:     {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 6, 7140, 32]} : tensor<1x42840x32xf16> -> tensor<1x6x7140x32xf16>
        // CHECK:       [[PERMUTEQUANTIZE:%.+]] = IE.PermuteQuantize([[AFFINERESHAPE1]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 10, 0, 0]} : tensor<1x6x7140x32xf16> -> tensor<1x16x7140x32xf16, {order = #NHWC}>
        // CHECK:       [[SLICE2:%.+]] = IE.Slice [[PERMUTEQUANTIZE]] [0, 0, 0, 0] [1, 16, 7140, 18] : tensor<1x16x7140x32xf16, {order = #NHWC}> to tensor<1x16x7140x18xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL1:%.+]] = IE.MaxPool([[SLICE2]]) {kernel_size = [1, 6], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 6]} : tensor<1x16x7140x18xf16, {order = #NHWC}> -> tensor<1x16x7140x3xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL2:%.+]] = IE.MaxPool([[MAXPOOL1]]) {kernel_size = [1, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x16x7140x3xf16, {order = #NHWC}> -> tensor<1x16x7140x1xf16>
        // CHECK:       [[SLICE3:%.+]] = IE.Slice [[MAXPOOL2]] [0, 0, 0, 0] [1, 6, 7140, 1] : tensor<1x16x7140x1xf16> to tensor<1x6x7140x1xf16
        // CHECK:       [[AFFINERESHAPE2:%.+]] = IE.AffineReshape([[SLICE3]])
        // CHECK-SAME{LITERAL}:        {dim_mapping = [[0], [1], [1], [2, 3]], shape_value = [1, 42840, 1, 1]} : tensor<1x6x7140x1xf16> -> tensor<1x42840x1x1xf16>
        // CHECK:       [[AFFINERESHAPE3:%.+]] = IE.AffineReshape([[AFFINERESHAPE2]])
        // CHECK-SAME{LITERAL}:        {dim_mapping = [[0], [1], [2], [2]], shape_value = [1, 42840, 1]} : tensor<1x42840x1x1xf16> -> tensor<1x42840x1xf16>
        // CHECK:       return [[AFFINERESHAPE3]] : tensor<1x42840x1xf16>
}
