//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie %s | FileCheck %s --strict-whitespace
// REQUIRES: platform-NPU4000

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
        // CHECK-SAME:      tensor<48x3x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]

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
// CHECK-DAG: [[Q_TYPE:!.+]] = !quant.uniform<!QuantileType.quantile<ui4:f16, {{.*}}>:f16, 2.000000e+00>
// CHECK-DAG: [[Q_TYPE1:!.+]] = !quant.uniform<u4:f16, 2.000000e+00:8>
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
        // CHECK-DAG:   [[WEIGHTS_0:%.+]] = const.Declare tensor<4096x1024x1x1x[[Q_TYPE]], {order = #NHWC}> = dense<1.000000e+00> : tensor<3x1024x4096xf32>, [#const.SubView<[0, 0, 0], [1, 1024, 4096]>, #const.Reshape<[1, 1024, 1, 4096]>, #const.CastElemType<f16>, #const.CastElemType<[[Q_TYPE1]]>, #const.Transpose<#NWHC>, #const.AffineReshape<{{\[\[}}0], [0], [0], [1, 2, 3]], [4096, 1024, 1, 1]>, #const.CastElemType<[[Q_TYPE]]>, #const.Reorder<#NHWC>]
        // CHECK-DAG:   [[WEIGHTS_1:%.+]] = const.Declare tensor<4096x1024x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x1024x4096xf32>, [#const.SubView<[1, 0, 0], [1, 1024, 4096]>, #const.Reshape<[1, 1024, 1, 4096]>, #const.CastElemType<f16>, #const.CastElemType<[[Q_TYPE1]]>, #const.Transpose<#NWHC>, #const.AffineReshape<{{\[\[}}0], [0], [0], [1, 2, 3]], [4096, 1024, 1, 1]>, #const.CastElemType<[[Q_TYPE]]>, #const.Reorder<#NHWC>]
        // CHECK-DAG:   [[WEIGHTS_2:%.+]] = const.Declare tensor<4096x1024x1x1x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<3x1024x4096xf32>, [#const.SubView<[2, 0, 0], [1, 1024, 4096]>, #const.Reshape<[1, 1024, 1, 4096]>, #const.CastElemType<f16>, #const.CastElemType<[[Q_TYPE1]]>, #const.Transpose<#NWHC>, #const.AffineReshape<{{\[\[}}0], [0], [0], [1, 2, 3]], [4096, 1024, 1, 1]>, #const.CastElemType<[[Q_TYPE]]>, #const.Reorder<#NHWC>]

        // CHECK:   [[RESHAPE_LHS:%.+]] = IE.AffineReshape([[ARG]]) {
        // CHECK-SAME:      shape_value = [1, 1, 16, 3072]
        // CHECK-SAME:  } : tensor<16x3072xf32> -> tensor<1x1x16x3072xf32>
        // CHECK:   [[CONVERT_LHS:%.+]] = IE.Convert([[RESHAPE_LHS]]) {
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

        %RESHAPE = IE.Reshape(%FQ) {shape_value = [3072, 4096]} : tensor<3x1024x4096xf32> -> tensor<3072x4096xf32>
        %GEMM = IE.MatMul(%arg0, %RESHAPE) : tensor<16x3072xf32>, tensor<3072x4096xf32> -> tensor<16x4096xf32>
        // CHECK:   [[RESHAPE_CONVERT:%.+]] = IE.AffineReshape([[CONVERT_LHS]]) {
        // CHECK-SAME:      shape_value = [16, 3072, 1, 1]
        // CHECK-SAME:  } : tensor<1x1x16x3072xf16> -> tensor<16x3072x1x1xf16>

        // CHECK:   [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[RESHAPE_CONVERT]]) {
        // CHECK-SAME:      dst_order = #NHWC,
        // CHECK-SAME:      mem_perm = #map
        // CHECK-SAME:  } : tensor<16x3072x1x1xf16> -> tensor<1x3072x16x1xf16, {order = #NHWC}>

        // CHECK-DAG:   [[SLICE_2:%.+]] = IE.Slice [[PERMUTE_CAST]] [0, 2048, 0, 0] [1, 1024, 16, 1] : tensor<1x3072x16x1xf16, {order = #NHWC}> to tensor<1x1024x16x1xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SLICE_1:%.+]] = IE.Slice [[PERMUTE_CAST]] [0, 1024, 0, 0] [1, 1024, 16, 1] : tensor<1x3072x16x1xf16, {order = #NHWC}> to tensor<1x1024x16x1xf16, {order = #NHWC}>
        // CHECK-DAG:   [[SLICE_0:%.+]] = IE.Slice [[PERMUTE_CAST]] [0, 0, 0, 0] [1, 1024, 16, 1] : tensor<1x3072x16x1xf16, {order = #NHWC}> to tensor<1x1024x16x1xf16, {order = #NHWC}>

        // CHECK:   [[CONV_INPUT_0:%.+]] = IE.AffineReshape([[SLICE_0]]) {
        // CHECK-SAME:      shape_value = [1, 1024, 4, 4]
        // CHECK-SAME:  } : tensor<1x1024x16x1xf16, {order = #NHWC}> -> tensor<1x1024x4x4xf16, {order = #NHWC}>
        // CHECK:   [[CONV_0:%.+]] = IE.Convolution([[CONV_INPUT_0]], [[WEIGHTS_0]]) {
        // CHECK-SAME:      dilations = [1, 1],
        // CHECK-SAME:      pads_begin = [0, 0],
        // CHECK-SAME:      pads_end = [0, 0],
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME:  } : tensor<1x1024x4x4xf16, {order = #NHWC}>, tensor<4096x1024x1x1x!qElemType, {order = #NHWC}> -> tensor<1x4096x4x4xf16, {order = #NHWC}>

        // CHECK:   [[CONV_INPUT_1:%.+]] = IE.AffineReshape([[SLICE_1]]) {
        // CHECK-SAME:      shape_value = [1, 1024, 4, 4]
        // CHECK-SAME:  } : tensor<1x1024x16x1xf16, {order = #NHWC}> -> tensor<1x1024x4x4xf16, {order = #NHWC}>
        // CHECK:   [[CONV_1:%.+]] = IE.Convolution([[CONV_INPUT_1]], [[WEIGHTS_1]]) {
        // CHECK-SAME:      dilations = [1, 1],
        // CHECK-SAME:      pads_begin = [0, 0],
        // CHECK-SAME:      pads_end = [0, 0],
        // CHECK-SAME:      strides = [1, 1]
        // CHECK-SAME:  } : tensor<1x1024x4x4xf16, {order = #NHWC}>, tensor<4096x1024x1x1x!qElemType, {order = #NHWC}> -> tensor<1x4096x4x4xf16, {order = #NHWC}>

        // CHECK:   [[CONV_INPUT_2:%.+]] = IE.AffineReshape([[SLICE_2]]) {
        // CHECK-SAME:      shape_value = [1, 1024, 4, 4]
        // CHECK-SAME:  } : tensor<1x1024x16x1xf16, {order = #NHWC}> -> tensor<1x1024x4x4xf16, {order = #NHWC}>
        // CHECK:   [[CONV_2:%.+]] = IE.Convolution([[CONV_INPUT_2]], [[WEIGHTS_2]]) {
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

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @DoNotUnrollMatMul
module @DoNotUnrollMatMul {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x6x64x24xf16>
        DataInfo "input" : tensor<1x6x64x24xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x6x64x64xf16>
    }

    // CHECK-LABEL: func.func @main
    // CHECK-SAME: [[ARG0:%.+]]: tensor<1x6x64x24xf16>, [[ARG1:%.+]]: tensor<1x6x64x24xf16>
    func.func @main(%arg0: tensor<1x6x64x24xf16>, %arg1: tensor<1x6x64x24xf16>) -> tensor<1x6x64x64xf16> {
        %0 = IE.MatMul(%arg0, %arg1) {transpose_b}
            : tensor<1x6x64x24xf16>, tensor<1x6x64x24xf16> -> tensor<1x6x64x64xf16>

        return %0 : tensor<1x6x64x64xf16>

        // CHECK:   [[EXPAND_WEIGHTS:%.+]] = const.Declare tensor<64x48x1x1xf16, {order = #NHWC}> = dense_resource<__elided__> : tensor<64x48x1x1xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
        // CHECK:   [[PERMUTECAST_0:%.+]] = IE.PermuteCast([[ARG0]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x6x64x24xf16> -> tensor<1x24x6x64xf16, {order = #NHWC}>
        // CHECK:   [[RESHAPE_0:%.+]] = IE.ShapeCast {shape = [1, 48, 6, 32]} inputs([[PERMUTECAST_0]] : tensor<1x24x6x64xf16, {order = #NHWC}>) -> tensor<1x48x6x32xf16, {order = #NHWC}>
        // CHECK:   [[CONV_0:%.+]] = IE.Convolution([[RESHAPE_0]], [[EXPAND_WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x48x6x32xf16, {order = #NHWC}>, tensor<64x48x1x1xf16, {order = #NHWC}> -> tensor<1x64x6x32xf16, {order = #NHWC}>
        // CHECK:   [[RESHAPE_1:%.+]] = IE.ShapeCast {shape = [1, 32, 6, 64]} inputs([[CONV_0]] : tensor<1x64x6x32xf16, {order = #NHWC}>) -> tensor<1x32x6x64xf16, {order = #NHWC}>
        // CHECK:   [[PERMUTECAST_1:%.+]] = IE.PermuteCast([[RESHAPE_1]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x32x6x64xf16, {order = #NHWC}> -> tensor<1x6x64x32xf16>
        // CHECK:   [[PERMUTECAST_2:%.+]] = IE.PermuteCast([[ARG1]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x6x64x24xf16> -> tensor<1x24x6x64xf16, {order = #NHWC}>
        // CHECK:   [[RESHAPE_2:%.+]] = IE.ShapeCast {shape = [1, 48, 6, 32]} inputs([[PERMUTECAST_2]] : tensor<1x24x6x64xf16, {order = #NHWC}>) -> tensor<1x48x6x32xf16, {order = #NHWC}>
        // CHECK:   [[CONV_1:%.+]] = IE.Convolution([[RESHAPE_2]], [[EXPAND_WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x48x6x32xf16, {order = #NHWC}>, tensor<64x48x1x1xf16, {order = #NHWC}> -> tensor<1x64x6x32xf16, {order = #NHWC}>
        // CHECK:   [[RESHAPE_3:%.+]] = IE.ShapeCast {shape = [1, 32, 6, 64]} inputs([[CONV_1]] : tensor<1x64x6x32xf16, {order = #NHWC}>) -> tensor<1x32x6x64xf16, {order = #NHWC}>
        // CHECK:   [[PERMUTECAST_3:%.+]] = IE.PermuteCast([[RESHAPE_3]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x32x6x64xf16, {order = #NHWC}> -> tensor<1x6x64x32xf16>
        // CHECK:   [[MATMUL:%.+]]= IE.MatMul([[PERMUTECAST_1]], [[PERMUTECAST_3]]) {transpose_b} : tensor<1x6x64x32xf16>, tensor<1x6x64x32xf16> -> tensor<1x6x64x64xf16>
        // CHECK:   return [[MATMUL]]
    }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @MemPermuteProcessingWithNDMemPermute
module @MemPermuteProcessingWithNDMemPermute {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<2x96x288x288xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<2x64x288x288xf16>
    }

    // CHECK: func.func @main([[ARG0:%.+]]: tensor<2x96x288x288xf16>) -> tensor<2x64x288x288xf16> {
    func.func @main(%arg0: tensor<2x96x288x288xf16>) -> tensor<2x64x288x288xf16> {
        %weights = const.Declare tensor<64x192x3x3xf16> = dense<1.0> : tensor<64x192x3x3xf16>
        %1 = IE.Concat(%arg0, %arg0) {static_offsets = [[0, 0, 0, 0], [0, 96, 0, 0]]} : tensor<2x96x288x288xf16>, tensor<2x96x288x288xf16> -> tensor<2x192x288x288xf16>

        %2 = IE.Convolution(%1, %weights) {
                dilations = [1, 1],
                pads_begin = [1, 1],
                pads_end = [1, 1],
                strides = [1, 1]
        } : tensor<2x192x288x288xf16>, tensor<64x192x3x3xf16>
            -> tensor<2x64x288x288xf16>

        %3 = IE.Sigmoid(%2) : tensor<2x64x288x288xf16> -> tensor<2x64x288x288xf16>

        return %3 : tensor<2x64x288x288xf16>

        // CHECK:       [[CST:%.+]] = const.Declare tensor<64x192x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x192x3x3xf16>, [#const.Reorder<#NHWC>]
        // CHECK:       [[CONCAT1:%.+]] = IE.Concat([[ARG0]], [[ARG0]])
        // CHECK-SAME{LITERAL}:      {static_offsets = [[0, 0, 0, 0], [0, 96, 0, 0]]} : tensor<2x96x288x288xf16>, tensor<2x96x288x288xf16> -> tensor<2x192x288x288xf16>
        // CHECK:       [[SLICE1:%.+]] = IE.Slice [[CONCAT1]] [0, 0, 0, 0] [1, 192, 288, 288] : tensor<2x192x288x288xf16> to tensor<1x192x288x288xf16>
        // CHECK:       [[PERMUTEQUANT1:%.+]] = IE.PermuteQuantize([[SLICE1]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x192x288x288xf16> -> tensor<1x192x288x288xf16, {order = #NHWC}>
        // CHECK:       [[SLICE2:%.+]] = IE.Slice [[CONCAT1]] [1, 0, 0, 0] [1, 192, 288, 288] : tensor<2x192x288x288xf16> to tensor<1x192x288x288xf16>
        // CHECK:       [[PERMUTEQUANT2:%.+]] = IE.PermuteQuantize([[SLICE2]]) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x192x288x288xf16> -> tensor<1x192x288x288xf16, {order = #NHWC}>
        // CHECK:       [[CONV1:%.+]] = IE.Convolution([[PERMUTEQUANT1]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x192x288x288xf16, {order = #NHWC}>, tensor<64x192x3x3xf16, {order = #NHWC}> -> tensor<1x64x288x288xf16, {order = #NHWC}>
        // CHECK:       [[CONV2:%.+]] = IE.Convolution([[PERMUTEQUANT2]], [[CST]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x192x288x288xf16, {order = #NHWC}>, tensor<64x192x3x3xf16, {order = #NHWC}> -> tensor<1x64x288x288xf16, {order = #NHWC}>
        // CHECK:       [[CONCAT2:%.+]] = IE.Concat([[CONV1]], [[CONV2]])
        // CHECK-SAME{LITERAL}:     {static_offsets = [[0, 0, 0, 0], [1, 0, 0, 0]]} : tensor<1x64x288x288xf16, {order = #NHWC}>, tensor<1x64x288x288xf16, {order = #NHWC}> -> tensor<2x64x288x288xf16, {order = #NHWC}>
        // CHECK:       [[SIGMOID:%.+]] = IE.Sigmoid([[CONCAT2]]) : tensor<2x64x288x288xf16, {order = #NHWC}> -> tensor<2x64x288x288xf16, {order = #NHWC}>
        // CHECK:       [[SHAPE_CAST1:%.+]] = IE.ShapeCast {shape = [1, 64, 2, 82944]} inputs([[SIGMOID]] : tensor<2x64x288x288xf16, {order = #NHWC}>) -> tensor<1x64x2x82944xf16, {order = #NHWC}>
        // CHECK:       [[MAXPOOL:%.+]] = IE.MaxPool([[SHAPE_CAST1]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x64x2x82944xf16, {order = #NHWC}> -> tensor<1x64x2x82944xf16, {order = #NHCW}>
        // CHECK:       [[PERMUTE_CAST:%.+]] = IE.PermuteCast([[MAXPOOL]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x2x82944xf16, {order = #NHCW}> -> tensor<1x2x64x82944xf16>
        // CHECK:       [[SHAPE_CAST2:%.+]] = IE.ShapeCast {shape = [2, 64, 288, 288]} inputs([[PERMUTE_CAST]] : tensor<1x2x64x82944xf16>) -> tensor<2x64x288x288xf16>
        // CHECK:       return [[SHAPE_CAST2]] : tensor<2x64x288x288xf16>
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

        // CHECK:   [[GROUP_0:%.+]] = IE.Add([[CONV_0]], [[CONV_1]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x91xf16, {order = #NHWC}>, tensor<1x64x1x91xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16, {order = #NCWH}>
        // CHECK:   [[PERMUTE_0:%.+]] = IE.PermuteCast([[GROUP_0]]) {
        // CHECK-SAME:      dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x64x1x91xf16, {order = #NCWH}> -> tensor<1x64x1x91xf16>

        // CHECK:   [[CONV_2:%.+]] = IE.Convolution([[SLICE_IN_2]], [[SLICE_WEIGHT_2]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x101xf16, {order = #NHWC}>, tensor<64x64x1x11xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16, {order = #NHWC}>
        // CHECK:   [[CONV_3:%.+]] = IE.Convolution([[SLICE_IN_3]], [[SLICE_WEIGHT_3]]) {
        // CHECK-SAME:      dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x1x101xf16, {order = #NHWC}>, tensor<64x64x1x11xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16, {order = #NHWC}>

        // CHECK:   [[GROUP_1:%.+]] = IE.Add([[CONV_2]], [[CONV_3]]) {auto_broadcast = #IE.auto_broadcast_type<NONE_OR_EXPLICIT>} : tensor<1x64x1x91xf16, {order = #NHWC}>, tensor<1x64x1x91xf16, {order = #NHWC}> -> tensor<1x64x1x91xf16, {order = #NCWH}>
        // CHECK:   [[PERMUTE_1:%.+]] = IE.PermuteCast([[GROUP_1]]) {
        // CHECK-SAME:      dst_order = #NCHW, mem_perm = #NCWH} : tensor<1x64x1x91xf16, {order = #NCWH}> -> tensor<1x64x1x91xf16>

        // CHECK:   [[CONCAT:%.+]] = IE.Concat([[PERMUTE_0]], [[PERMUTE_1]]) {
        // CHECK-SAME{LITERAL}:      static_offsets = [[0, 0, 0, 0], [0, 64, 0, 0]]} : tensor<1x64x1x91xf16>, tensor<1x64x1x91xf16> -> tensor<1x128x1x91xf16>
        // CHECK:   return [[CONCAT]] : tensor<1x128x1x91xf16>
    }
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>

// CHECK-LABEL: @PropagateMemPermuteMultiplyAdd
module @PropagateMemPermuteMultiplyAdd {

net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input0" : tensor<1x4x256x1xf16, {order = #NHWC}>
        DataInfo "input1" : tensor<4x4x1x1xf16, {order = #NHWC}>
        DataInfo "input2" : tensor<1x4x1x1xf16>
        DataInfo "input3" : tensor<1x2048x256x1xf16, {order = #NHWC}>
        DataInfo "input4" : tensor<1x4x256x2048xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<4x1x256x2048xf16>
    }

    // CHECK-LABEL: func.func @main
    // CHECK-SAME:      [[INPUT_0:%.+]]: tensor<1x4x256x1xf16, {order = #NHWC}>,
    // CHECK-SAME:      [[INPUT_1:%.+]]: tensor<4x4x1x1xf16, {order = #NHWC}>,
    // CHECK-SAME:      [[INPUT_2:%.+]]: tensor<1x4x1x1xf16>,
    // CHECK-SAME:      [[INPUT_3:%.+]]: tensor<1x2048x256x1xf16, {order = #NHWC}>,
    // CHECK-SAME:      [[INPUT_4:%.+]]: tensor<1x4x256x2048xf16>)
    func.func @main(%arg0: tensor<1x4x256x1xf16, {order = #NHWC}>,
                    %arg1: tensor<4x4x1x1xf16, {order = #NHWC}>,
                    %arg2: tensor<1x4x1x1xf16>,
                    %arg3: tensor<1x2048x256x1xf16, {order = #NHWC}>,
                    %arg4: tensor<1x4x256x2048xf16>) -> tensor<4x1x256x2048xf16> {

        %0 = IE.Convolution(%arg0, %arg1, %arg2) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x4x256x1xf16, {order = #NHWC}>, tensor<4x4x1x1xf16, {order = #NHWC}>, tensor<1x4x1x1xf16> -> tensor<1x4x256x1xf16, {order = #NHWC}>
        %1 = IE.PermuteCast(%arg3) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x2048x256x1xf16, {order = #NHWC}> -> tensor<1x256x1x2048xf16>
        %2 = IE.AffineReshape(%1) {dim_mapping = [[0, 1], [2], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x1x2048xf16> -> tensor<1x1x256x2048xf16>
        %3 = IE.PermuteCast(%2) {dst_order = #NHWC, mem_perm = #NHWC} : tensor<1x1x256x2048xf16> -> tensor<1x1x256x2048xf16, {order = #NHWC}>
        %4 = IE.Multiply(%0, %3) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x256x1xf16, {order = #NHWC}>, tensor<1x1x256x2048xf16, {order = #NHWC}> -> tensor<1x4x256x2048xf16, {order = #NHWC}>
        %5 = IE.PermuteQuantize(%arg4) {dstElemType = f16, dst_order = #NHWC, mem_perm = #NHWC, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]} : tensor<1x4x256x2048xf16> -> tensor<1x4x256x2048xf16, {order = #NHWC}>
        %6 = IE.Add(%5, %4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x256x2048xf16, {order = #NHWC}>, tensor<1x4x256x2048xf16, {order = #NHWC}> -> tensor<1x4x256x2048xf16, {order = #NHWC}>
        %7 = IE.MemPermute(%6) {dst_order = #NCHW, mem_perm = #map1} : tensor<1x4x256x2048xf16, {order = #NHWC}> -> tensor<4x1x256x2048xf16>

        return %7 : tensor<4x1x256x2048xf16>

        // CHECK-DAG:   [[CST:%.+]] = const.Declare tensor<64x16x1x1xf16, {order = #NHWC}> = dense_resource<__elided__> : tensor<64x16x1x1xf16>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>]
        // CHECK-DAG:   [[CST_0:%.+]] = const.Declare tensor<4x12x1x1xf16, {order = #NHWC}> = dense<0.000000e+00> : tensor<4x12x1x1xf16, {order = #NHWC}>
        // CHECK:       [[AFFINE_RESHAPE_0:%.+]] = IE.ShapeCast {shape = [1, 16, 64, 1]} inputs([[INPUT_0]] : tensor<1x4x256x1xf16, {order = #NHWC}>) -> tensor<1x16x64x1xf16, {order = #NHWC}>
        // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[AFFINE_RESHAPE_0]], [[CST]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x64x1xf16, {order = #NHWC}>, tensor<64x16x1x1xf16, {order = #NHWC}> -> tensor<1x64x64x1xf16, {order = #NHWC}>
        // CHECK:       [[AFFINE_RESHAPE_1:%.+]] = IE.ShapeCast {shape = [1, 16, 256, 1]} inputs([[CONV_0]] : tensor<1x64x64x1xf16, {order = #NHWC}>) -> tensor<1x16x256x1xf16, {order = #NHWC}>
        // CHECK:       [[CONCAT:%.+]] = IE.Concat([[INPUT_1]], [[CST_0]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 4, 0, 0]]} : tensor<4x4x1x1xf16, {order = #NHWC}>, tensor<4x12x1x1xf16, {order = #NHWC}> -> tensor<4x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[EXPAND_0:%.+]] = IE.Expand([[CONCAT]]) {pads_begin = [0, 0, 0, 0], pads_end = [12, 0, 0, 0]} : tensor<4x16x1x1xf16, {order = #NHWC}> -> tensor<16x16x1x1xf16, {order = #NHWC}>
        // CHECK:       [[EXPAND_1:%.+]] = IE.Expand([[INPUT_2]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 12, 0, 0]} : tensor<1x4x1x1xf16> -> tensor<1x16x1x1xf16>
        // CHECK:       [[AFFINE_RESHAPE_2:%.+]] = IE.AffineReshape([[AFFINE_RESHAPE_1]]) {dim_mapping = {{\[\[}}0], [1], [2, 3], [3]], shape_value = [1, 16, 64, 4]} : tensor<1x16x256x1xf16, {order = #NHWC}> -> tensor<1x16x64x4xf16, {order = #NHWC}>
        // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[AFFINE_RESHAPE_2]], [[EXPAND_0]], [[EXPAND_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x64x4xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}>, tensor<1x16x1x1xf16> -> tensor<1x16x64x4xf16>
        // CHECK:       [[SLICE:%.+]] = IE.Slice [[CONV_1]] [0, 0, 0, 0] [1, 4, 64, 4] : tensor<1x16x64x4xf16> to tensor<1x4x64x4xf16>
        // CHECK:       [[PERMUTE_CAST_0:%.+]] = IE.PermuteCast([[INPUT_3]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x2048x256x1xf16, {order = #NHWC}> -> tensor<1x256x1x2048xf16>
        // CHECK:       [[AFFINE_RESHAPE_3:%.+]] = IE.AffineReshape([[PERMUTE_CAST_0]]) {dim_mapping = {{\[\[}}0, 1], [2], [2], [3]], shape_value = [1, 1, 256, 2048]} : tensor<1x256x1x2048xf16> -> tensor<1x1x256x2048xf16>
        // CHECK:       [[AFFINE_RESHAPE_4:%.+]] = IE.AffineReshape([[SLICE]]) {dim_mapping = {{\[\[}}0], [1], [2], [2, 3]], shape_value = [1, 4, 256, 1]} : tensor<1x4x64x4xf16> -> tensor<1x4x256x1xf16>
        // CHECK:       [[MULTIPLY:%.+]] = IE.Multiply([[AFFINE_RESHAPE_4]], [[AFFINE_RESHAPE_3]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x4x256x1xf16>, tensor<1x1x256x2048xf16> -> tensor<1x4x256x2048xf16>
        // CHECK:       [[PERMUTE_CAST_1:%.+]] = IE.PermuteCast([[MULTIPLY]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x256x2048xf16> -> tensor<1x2048x4x256xf16, {order = #NHWC}>
        // CHECK:       [[PERMUTE_CAST_2:%.+]] = IE.PermuteCast([[INPUT_4]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x4x256x2048xf16> -> tensor<1x2048x4x256xf16, {order = #NHWC}>
        // CHECK:       [[SHAPE_CAST_0:%.+]] = IE.ShapeCast {shape = [1, 2048, 256, 4]} inputs([[PERMUTE_CAST_2]] : tensor<1x2048x4x256xf16, {order = #NHWC}>) -> tensor<1x2048x256x4xf16, {order = #NHWC}>
        // CHECK:       [[SHAPE_CAST_1:%.+]] = IE.ShapeCast {shape = [1, 2048, 256, 4]} inputs([[PERMUTE_CAST_1]] : tensor<1x2048x4x256xf16, {order = #NHWC}>) -> tensor<1x2048x256x4xf16, {order = #NHWC}>
        // CHECK:       [[ADD:%.+]] = IE.Add([[SHAPE_CAST_0]], [[SHAPE_CAST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2048x256x4xf16, {order = #NHWC}>, tensor<1x2048x256x4xf16, {order = #NHWC}> -> tensor<1x2048x256x4xf16, {order = #NHWC}>
        // CHECK:       [[SHAPE_CAST_2:%.+]] = IE.ShapeCast {shape = [4, 2048, 1, 256]} inputs([[ADD]] : tensor<1x2048x256x4xf16, {order = #NHWC}>) -> tensor<4x2048x1x256xf16, {order = #NHWC}>
        // CHECK:       [[PERMUTE_CAST_3:%.+]] = IE.PermuteCast([[SHAPE_CAST_2]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<4x2048x1x256xf16, {order = #NHWC}> -> tensor<4x1x256x2048xf16>
        // CHECK:       return [[PERMUTE_CAST_3]] : tensor<4x1x256x2048xf16>
    }
}
