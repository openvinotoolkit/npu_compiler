//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="extraction-mode=gen-all" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// CHECK-LABEL: @CommonSubexpressionElimination
{-#
    dialect_resources: {
        builtin: {
            ov_1: "0x0000000400aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbdd",
            ov_2: "0x0000000400aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbdd"
        }
    }
#-}

module @CommonSubexpressionElimination {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<4x4xf32>
        DataInfo "output2" : tensor<4x4xf32>
        DataInfo "output3" : tensor<8x4xf32>
        DataInfo "output4" : tensor<8x4xf16>
        DataInfo "output5" : tensor<8x4xf16>
        DataInfo "output6" : tensor<4x4xf32>
        DataInfo "output7" : tensor<4x4xf32>
    }

    func.func @main() -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<8x4xf32>, tensor<8x4xf16>, tensor<8x4xf16>, tensor<4x4xf32>, tensor<4x4xf32>) {
        %cst_t1 = const.Declare tensor<4x4xf32> = dense_resource<ov_1> : tensor<4x4xf32>, [#const.Add<1.0 : f32>]
        %cst_t2 = const.Declare tensor<4x4xf32> = dense_resource<ov_2> : tensor<4x4xf32>, [#const.Add<1.0 : f32>]
        %cst_t2_t3_t4 = const.Declare tensor<8x4xf32> = dense_resource<ov_2> : tensor<4x4xf32>, [#const.Add<1.0 : f32>, #const.PadWithZero<[0, 0], [4, 0]>, #const.Rescale<5.0 : f32>]
        %cst_t2_t3_t5 = const.Declare tensor<8x4xf16> = dense_resource<ov_2> : tensor<4x4xf32>, [#const.Add<1.0 : f32>, #const.PadWithZero<[0, 0], [4, 0]>, #const.ConvertElemType<f16>]
        %cst_t2_t3_t5_copy = const.Declare tensor<8x4xf16> = dense_resource<ov_2> : tensor<4x4xf32>, [#const.Add<1.0 : f32>, #const.PadWithZero<[0, 0], [4, 0]>, #const.ConvertElemType<f16>]
        %cst_empty_1 = const.Declare tensor<4x4xf32> = dense_resource<ov_2> : tensor<4x4xf32>
        %cst_empty_2 = const.Declare tensor<4x4xf32> = dense_resource<ov_2> : tensor<4x4xf32>, []
        return %cst_t1, %cst_t2, %cst_t2_t3_t4, %cst_t2_t3_t5, %cst_t2_t3_t5_copy, %cst_empty_1, %cst_empty_2 : tensor<4x4xf32>, tensor<4x4xf32>, tensor<8x4xf32>, tensor<8x4xf16>, tensor<8x4xf16>, tensor<4x4xf32>, tensor<4x4xf32>
    }

    // CHECK:       IE.CNNNetwork entryPoint : @wrapper_main inputsInfo : {
    // CHECK:       } outputsInfo : {
    // CHECK:           DataInfo "output1" : tensor<4x4xf32>

    // CHECK:       func.func private @init([[NGRAPH_1:%.+]]: tensor<4x4xf32>, [[NGRAPH_2:%.+]]: tensor<4x4xf32>)
    // CHECK-SAME:          -> (tensor<4x4xf32>, tensor<8x4xf32>, tensor<8x4xf16>, tensor<4x4xf32>)
    // CHECK:           [[CST_0:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
    // CHECK:           [[CST_T1:%.+]] = IE.Add([[NGRAPH_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4xf32>, tensor<1xf32> -> tensor<4x4xf32>
    // CHECK:           [[CST_1:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
    // CHECK:           [[CST_T2:%.+]] = IE.Add([[NGRAPH_2]], [[CST_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4xf32>, tensor<1xf32> -> tensor<4x4xf32>
    // CHECK:           [[CST_T2_T3:%.+]] = IE.Pad([[CST_T2]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0], pads_end_attr = [4, 0]} : tensor<4x4xf32> -> tensor<8x4xf32>
    // CHECK:           [[CST_2:%.+]] = const.Declare tensor<1xf32> = dense<5.000000e+00> : tensor<1xf32>
    // CHECK:           [[CST_T2_T3_T4:%.+]] = IE.Multiply([[CST_T2_T3]], [[CST_2]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x4xf32>, tensor<1xf32> -> tensor<8x4xf32>
    // CHECK:           [[CST_T2_T3_T5:%.+]] = IE.Convert([[CST_T2_T3]]) {dstElemType = f16} : tensor<8x4xf32> -> tensor<8x4xf16>
    // CHECK:           return [[CST_T1]], [[CST_T2_T3_T4]], [[CST_T2_T3_T5]], [[CST_T2]]

    // CHECK:       func.func private @main([[ARG0:%.+]]: tensor<4x4xf32>, [[ARG2:%.+]]: tensor<8x4xf32>, [[ARG3:%.+]]: tensor<8x4xf16>, [[ARG1:%.+]]: tensor<4x4xf32>)
    // CHECK-SAME:          -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<8x4xf32>, tensor<8x4xf16>, tensor<8x4xf16>, tensor<4x4xf32>, tensor<4x4xf32>)
    // CHECK:           [[CST2:%.+]] = const.Declare tensor<4x4xf32> = dense_resource<ov_2> : tensor<4x4xf32>
    // CHECK:           [[CST3:%.+]] = const.Declare tensor<4x4xf32> = dense_resource<ov_2> : tensor<4x4xf32>
    // CHECK:           return [[ARG0]], [[ARG1]], [[ARG2]], [[ARG3]], [[ARG3]], [[CST2]], [[CST3]] : tensor<4x4xf32>, tensor<4x4xf32>, tensor<8x4xf32>, tensor<8x4xf16>, tensor<8x4xf16>, tensor<4x4xf32>, tensor<4x4xf32>

    // CHECK:       func.func @wrapper_main() -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<8x4xf32>, tensor<8x4xf16>, tensor<8x4xf16>, tensor<4x4xf32>, tensor<4x4xf32>)
    // CHECK:           [[CST0:%.+]] = const.Declare tensor<4x4xf32> = dense_resource<ov_1> : tensor<4x4xf32>
    // CHECK:           [[CST1:%.+]] = const.Declare tensor<4x4xf32> = dense_resource<ov_2> : tensor<4x4xf32>
    // CHECK:           [[CALL:%.+]]:4 = call @init([[CST0]], [[CST1]])
    // CHECK:           [[RET:%.+]]:7 = call @main([[CALL]]#0, [[CALL]]#1, [[CALL]]#2, [[CALL]]#3)
    // CHECK:           return [[RET]]#0, [[RET]]#1, [[RET]]#2, [[RET]]#3, [[RET]]#4, [[RET]]#5, [[RET]]#6
}

// -----

// CHECK-LABEL: @SubViewOutside
{-#
    dialect_resources: {
        builtin: {
            ov_1: "0x0000000400aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbdd",
            ov_2: "0x0000000400aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbdd"
        }
    }
#-}

module @SubViewOutside {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x2xf32>
    }

    func.func @main() -> (tensor<2x2xf32>) {
        %cst_t1 = const.Declare tensor<2x2xf32> = dense_resource<ov_1> : tensor<4x4xf32>, [#const.Add<1.0 : f32>, #const.SubView<[2, 2], [2, 2]>]
        return %cst_t1 : tensor<2x2xf32>
    }

    // CHECK: IE.CNNNetwork entryPoint : @wrapper_main inputsInfo : {
    // CHECK: } outputsInfo : {
    // CHECK:     DataInfo "output1" : tensor<2x2xf32>

    // CHECK: func.func private @init([[NGRAPH_1:%.+]]: tensor<4x4xf32>) -> tensor<4x4xf32>
    // CHECK:     [[CST_0:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>
    // CHECK:     [[CST_T1:%.+]] = IE.Add([[NGRAPH_1]], [[CST_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x4xf32>, tensor<1xf32> -> tensor<4x4xf32>
    // CHECK:     return [[CST_T1]] : tensor<4x4xf32>

    // CHECK: func.func private @main([[ARG0:%.+]]: tensor<4x4xf32>) -> tensor<2x2xf32>
    // CHECK:     [[SLICE:%.+]] = VPU.Slice [[ARG0]] [2, 2] [2, 2] : tensor<4x4xf32> to tensor<2x2xf32>
    // CHECK:     return [[SLICE]] : tensor<2x2xf32>

    // CHECK: func.func @wrapper_main() -> tensor<2x2xf32>
    // CHECK:     [[CST0:%.+]] = const.Declare tensor<4x4xf32> = dense_resource<ov_1> : tensor<4x4xf32>
    // CHECK:     [[CALL:%.+]] = call @init([[CST0]]) : (tensor<4x4xf32>) -> tensor<4x4xf32>
    // CHECK:     [[RET:%.+]] = call @main([[CALL]])
    // CHECK:     return [[RET]] : tensor<2x2xf32>
}

// -----

{-#
  dialect_resources: {
    builtin: {
      ov: "0x10000000ABABABABCDCDCDCD"
    }
  }
#-}

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @SubViewOutsideAdvanced
module @SubViewOutsideAdvanced {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "Parameter_58" : tensor<1x192x100x100xf16>
    } outputsInfo : {
        DataInfo "Convolution_63" friendlyName = "Result_64" : tensor<48x16x1x1xf16>
    }

    // CHECK: IE.CNNNetwork entryPoint : @wrapper_main inputsInfo : {
    // CHECK:     DataInfo "Parameter_58" : tensor<1x192x100x100xf16>
    // CHECK: } outputsInfo : {
    // CHECK:     DataInfo "Convolution_63" friendlyName = "Result_64" : tensor<48x16x1x1xf16>

    func.func @main(%arg0: tensor<1x192x100x100xf16>) -> tensor<48x16x1x1xf16> {
        %cst = const.Declare tensor<48x16x1x1xf16, {order = #NHWC}> = dense_resource<ov> : tensor<2x2x1x1xf16>, [#const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [46, 14, 0, 0]>, #const.SubView<[0, 0, 0, 0], [48, 16, 1, 1]>]
        %12 = VPU.NCE.ClusterTiling (%cst as %arg1: tensor<48x16x1x1xf16, {order = #NHWC}>) -> !VPU.DistributedTensor<48x16x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 4 : i64, uniform_distributed_segments, compute_shapes = [[48, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], memory_shapes = [[48, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1], [48, 16, 1, 1]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}> {
          %55 = VPU.Copy(%arg1) {out_mem_space = @CMX_NN} : tensor<48x16x1x1xf16, {order = #NHWC}> -> tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NHWC}>
          VPU.Yield %55
        }
        %13 = VPU.NCE.ClusterTiling (%12 as %arg1: tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<48x16x1x1xf16> {
          %55 = VPU.Copy(%arg1) : tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}> -> tensor<48x16x1x1xf16>
          VPU.Yield %55
        }
        return %13 : tensor<48x16x1x1xf16>
    }

    // CHECK:   func.func private @init([[OV_CONST0:%.+]]: tensor<2x2x1x1xf16>) -> tensor<48x16x1x1xf16, {order = #NHWC}>
    // CHECK:       [[REORDER:%.+]] = IE.Reorder([[OV_CONST0]]) {dstOrder = #NHWC} : tensor<2x2x1x1xf16> -> tensor<2x2x1x1xf16, {order = #NHWC}>
    // CHECK:       [[PAD:%.+]] = IE.Pad([[REORDER]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [46, 14, 0, 0]} : tensor<2x2x1x1xf16, {order = #NHWC}> -> tensor<48x16x1x1xf16, {order = #NHWC}>
    // CHECK:       return [[PAD]] : tensor<48x16x1x1xf16, {order = #NHWC}>

    // CHECK: func.func private @main([[ARG0:%.+]]: tensor<1x192x100x100xf16>, [[OVARG0:%.+]]: tensor<48x16x1x1xf16, {order = #NHWC}>) -> tensor<48x16x1x1xf16>
    // -- Ensure that the #const.SubViews have been converted to VPU ops.
    // CHECK:     [[SLICE:%.+]] = VPU.Slice [[OVARG0]] [0, 0, 0, 0] [48, 16, 1, 1] : tensor<48x16x1x1xf16, {order = #NHWC}> to tensor<48x16x1x1xf16, {order = #NHWC}>
    // CHECK:     [[TILING0:%.+]] = VPU.NCE.ClusterTiling ([[SLICE]] as {{%.+}}: tensor<48x16x1x1xf16, {order = #NHWC}>) -> !VPU.DistributedTensor
    // CHECK:     [[TILING1:%.+]] = VPU.NCE.ClusterTiling ([[TILING0]] as {{%.+}}: tensor<48x16x1x1xf16, {mem_space = @CMX_NN, order = #NCHW}>) -> tensor<48x16x1x1xf16> {
    // CHECK:     return [[TILING1]] : tensor<48x16x1x1xf16>

    // CHECK: func.func @wrapper_main([[ARG0:%.+]]: tensor<1x192x100x100xf16>) -> tensor<48x16x1x1xf16>
    // -- Ensure that the stripped ngraph constants are outside.
    // CHECK-DAG: [[CST0:%.+]] = const.Declare tensor<2x2x1x1xf16> = dense_resource<ov> : tensor<2x2x1x1xf16>
    // CHECK:     [[CALL:%.+]] = call @init([[CST0]]) : (tensor<2x2x1x1xf16>) -> tensor<48x16x1x1xf16, {order = #NHWC}>
    // CHECK:     [[RET:%.+]] = call @main([[ARG0]], [[CALL]])
    // CHECK:     return [[RET:%.+]] : tensor<48x16x1x1xf16>
}

// -----

{-#
  dialect_resources: {
    builtin: {
        ov_0: "0x10000000AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30"
    }
  }
#-}

!qElemType1 = !quant.uniform<i8:f16, 0.5>
// CHECK-DAG: [[QTYPE1:!.+]] = !quant.uniform<i8:f16, 5.000000e-01>

!qElemType2 = !quant.uniform<u8:f16, 0.5:128>
// CHECK-DAG: [[QTYPE2:!.+]] = !quant.uniform<u8:f16, 5.000000e-01:128>

// Note: CHECK-LABEL must NOT be used: it resets quantization checks above such
//       that [[QTYPE*]] captured variables become undefined.

// CHECK: module @QuantizedToQuantizedConversion
module @QuantizedToQuantizedConversion {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output_0" : tensor<16x3x3x3xui8>
    }

    // CHECK:    IE.CNNNetwork entryPoint : @wrapper_main inputsInfo : {
    // CHECK:    } outputsInfo : {
    // CHECK:        DataInfo "output_0" : tensor<16x3x3x3xui8>

    func.func @main() -> (tensor<16x3x3x3xui8>) {
        %cst = const.Declare tensor<16x3x3x3x!qElemType2> = dense_resource<ov_0> : tensor<16x3x3x3xsi8>,
            [#const.CastElemType<!qElemType1>, #const.ConvertElemType<!qElemType2>]

        // Normally QuantizeCast ops are part of transformations
        // But since network quantized I/O is not supported we add them here manually
        // Imagine Conv ops instead
        %0 = VPU.QuantizeCast(%cst) { dstElemType = ui8 }
            : tensor<16x3x3x3x!qElemType2> -> tensor<16x3x3x3xui8>

        return %0 : tensor<16x3x3x3xui8>
    }

    // CHECK: func.func private @init([[ARG0:%.+]]: tensor<16x3x3x3xsi8>)
    // CHECK:   [[CAST0:%.+]] = IE.QuantizeCast([[ARG0]]) {dstElemType = [[QTYPE1]]}
    // CHECK:   [[AVGPOOL0:%.+]] = IE.AvgPool([[CAST0]])
    // CHECK-SAME{LITERAL}: {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME{LITERAL}:  rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME: tensor<16x3x3x3x[[QTYPE1]]> -> tensor<16x3x3x3x[[QTYPE2]]>

    // CHECK:   [[BOUNDARY_CAST0:%.+]] = IE.QuantizeCast([[AVGPOOL0]]) {dstElemType = ui8}
    // CHECK:   return [[BOUNDARY_CAST0]]

    // CHECK: func.func private @main
    // CHECK-SAME: ([[INIT_OUT0:%.+]]: tensor<16x3x3x3xui8>)
    // CHECK:   [[BOUNDARY_CAST1:%.+]] = VPU.QuantizeCast([[INIT_OUT0]]) {dstElemType = [[QTYPE2]]}

    // CHECK:   [[RES:%.+]] = VPU.QuantizeCast([[BOUNDARY_CAST1]]) {dstElemType = ui8}
    // CHECK:   return [[RES]]

    // CHECK: func.func @wrapper_main() -> tensor<16x3x3x3xui8>
    // CHECK:   [[CST0:%.+]] = const.Declare tensor<16x3x3x3xsi8> = dense_resource<ov_0>
    // CHECK:   [[INIT:%.+]] = call @init([[CST0]])
    // CHECK:   [[RET:%.+]] = call @main([[INIT]])
    // CHECK:   return [[RET]]
}

// -----

{-#
  dialect_resources: {
    builtin: {
        ov_0: "0x10000000AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30",
        ov_1: "0x100000000ABDCE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE300AB0CE300AB0CE30CE30"
    }
  }
#-}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType1 = !quant.uniform<i8:f16:1, {8.9925130208333328E-4, 5.9925130208333328E-4, 6.9925130208333328E-4}>
// CHECK-DAG: [[QTYPE1:!.+]] = !quant.uniform<i8:f16:1, {8.9925130208333328E-4,5.9925130208333325E-4,6.992513020833333E-4}>
// CHECK-DAG: [[I8_PER_TENSOR:!.+]] = !quant.uniform<i8:f16, 1.000000e+00>
// CHECK-DAG: [[U8_PER_TENSOR:!.+]] = !quant.uniform<u8:f16, 1.000000e+00:128>

!qElemType2 = !quant.uniform<u8:f16:1, {8.9925130208333328E-4:128, 5.9925130208333328E-4:128, 6.9925130208333328E-4:128}>
// CHECK-DAG: [[QTYPE2:!.+]] = !quant.uniform<u8:f16:1, {8.9925130208333328E-4:128,5.9925130208333325E-4:128,6.992513020833333E-4:128}>

!qElemType3 = !quant.uniform<u8:f16:1, {8.9925130208333328E-4:128}>
// CHECK-DAG: [[QTYPE3:!.+]] = !quant.uniform<u8:f16:1, {8.9925130208333328E-4:128}>

!qElemType4 = !quant.uniform<u8:f16:1, {5.9925130208333328E-4:128, 6.9925130208333328E-4:128}>
// CHECK-DAG: [[QTYPE4:!.+]] = !quant.uniform<u8:f16:1, {5.9925130208333325E-4:128,6.992513020833333E-4:128}>

!qElemType5 = !quant.uniform<i8:f16:1, {8.9925130208333328E-4, 5.9925130208333328E-4, 6.9925130208333328E-4, 8.9925130208333328E-4, 5.9925130208333328E-4, 6.9925130208333328E-4, 8.9925130208333328E-4, 5.9925130208333328E-4, 6.9925130208333328E-4, 8.9925130208333328E-4}>
// CHECK-DAG: [[QTYPE5:!.+]] = !quant.uniform<i8:f16:1, {8.9925130208333328E-4,5.9925130208333325E-4,6.992513020833333E-4,8.9925130208333328E-4,5.9925130208333325E-4,6.992513020833333E-4,8.9925130208333328E-4,5.9925130208333325E-4,6.992513020833333E-4,8.9925130208333328E-4}>

!qElemType6 = !quant.uniform<i8:f16:0, {8.9925130208333328E-4, 5.9925130208333328E-4, 6.9925130208333328E-4, 8.9925130208333328E-4, 5.9925130208333328E-4, 6.9925130208333328E-4, 8.9925130208333328E-4, 5.9925130208333328E-4, 6.9925130208333328E-4, 8.9925130208333328E-4}>
// CHECK-DAG: [[QTYPE6:!.+]] = !quant.uniform<i8:f16:0, {8.9925130208333328E-4,5.9925130208333325E-4,6.992513020833333E-4,8.9925130208333328E-4,5.9925130208333325E-4,6.992513020833333E-4,8.9925130208333328E-4,5.9925130208333325E-4,6.992513020833333E-4,8.9925130208333328E-4}>

!qElemType7 = !quant.uniform<u8:f16:0, {8.9925130208333328E-4:128, 5.9925130208333328E-4:128, 6.9925130208333328E-4:128, 8.9925130208333328E-4:128, 5.9925130208333328E-4:128, 6.9925130208333328E-4:128, 8.9925130208333328E-4:128, 5.9925130208333328E-4:128, 6.9925130208333328E-4:128, 8.9925130208333328E-4:128}>
// CHECK-DAG: [[QTYPE7:!.+]] = !quant.uniform<u8:f16:0, {8.9925130208333328E-4:128,5.9925130208333325E-4:128,6.992513020833333E-4:128,8.9925130208333328E-4:128,5.9925130208333325E-4:128,6.992513020833333E-4:128,8.9925130208333328E-4:128,5.9925130208333325E-4:128,6.992513020833333E-4:128,8.9925130208333328E-4:128}>

// Note: CHECK-LABEL must NOT be used: it resets quantization checks above such
//       that [[QTYPE*]] captured variables become undefined.

// CHECK: module @QuantizedToQuantizedConversion_PerAxis
module @QuantizedToQuantizedConversion_PerAxis {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output_0" : tensor<16x1x3x3xui8, {order = #NHWC}>
        DataInfo "output_1" : tensor<16x2x3x3xui8, {order = #NHWC}>
        DataInfo "output_2" : tensor<10x20x1x1xui8>
    }

    // CHECK:    IE.CNNNetwork entryPoint : @wrapper_main inputsInfo : {
    // CHECK:    } outputsInfo : {
    // CHECK:        DataInfo "output_0" : tensor<16x1x3x3xui8, {order = #NHWC}>
    // CHECK:        DataInfo "output_1" : tensor<16x2x3x3xui8, {order = #NHWC}>
    // CHECK:        DataInfo "output_2" : tensor<10x20x1x1xui8>

    func.func @main() -> (tensor<16x1x3x3xui8, {order = #NHWC}>,  tensor<16x2x3x3xui8, {order = #NHWC}>, tensor<10x20x1x1xui8>) {
        %cst_0 = const.Declare tensor<16x1x3x3x!qElemType3, {order = #NHWC}> = dense_resource<ov_0> : tensor<16x3x3x3xsi8>,
                    [#const.CastElemType<!qElemType1>,
                    #const.ConvertElemType<!qElemType2>,
                    #const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [16, 1, 3, 3]>]
        %cst_1 = const.Declare tensor<16x2x3x3x!qElemType4, {order = #NHWC}> = dense_resource<ov_0> : tensor<16x3x3x3xsi8>,
                    [#const.CastElemType<!qElemType1>,
                    #const.ConvertElemType<!qElemType2>,
                    #const.Reorder<#NHWC>, #const.SubView<[0, 1, 0, 0], [16, 2, 3, 3]>]

        %cst_2 = const.Declare tensor<10x20x1x1x!qElemType7> = dense_resource<ov_1> : tensor<10x20xsi8>,
                    [#const.Reshape<[1, 10, 1, 20]>, #const.CastElemType<!qElemType5>,
                    #const.ChangeShapeAndElemType<[10, 20, 1, 1], !qElemType6>,
                    #const.ConvertElemType<!qElemType7>]

        // Normally QuantizeCast ops are part of transformations
        // But since network quantized I/O is not supported we add them here manually
        // Imagine Conv ops instead
        %0 = VPU.QuantizeCast(%cst_0) { dstElemType = ui8 }
                : tensor<16x1x3x3x!qElemType3, {order = #NHWC}> -> tensor<16x1x3x3xui8, {order = #NHWC}>

        %1 = VPU.QuantizeCast(%cst_1) { dstElemType = ui8 }
                : tensor<16x2x3x3x!qElemType4, {order = #NHWC}> -> tensor<16x2x3x3xui8, {order = #NHWC}>

        %2 = VPU.QuantizeCast(%cst_2) { dstElemType = ui8 }
                : tensor<10x20x1x1x!qElemType7> -> tensor<10x20x1x1xui8>

        return %0, %1, %2 : tensor<16x1x3x3xui8, {order = #NHWC}>,  tensor<16x2x3x3xui8, {order = #NHWC}>, tensor<10x20x1x1xui8>
    }

    // CHECK: func.func private @init([[ARG0:%.+]]: tensor<16x3x3x3xsi8>, [[ARG1:%.+]]: tensor<10x20xsi8>)
    // CHECK:   [[CAST0:%.+]] = IE.QuantizeCast([[ARG0]]) {dstElemType = [[QTYPE1]]}
    // CHECK:   [[CAST0_NORMALIZE:%.+]] = IE.QuantizeCast([[CAST0]]) {dstElemType = si8}
    // CHECK:   [[CAST0_PER_TENSOR:%.+]] = IE.QuantizeCast([[CAST0_NORMALIZE]]) {dstElemType = [[I8_PER_TENSOR]]}
    // CHECK:   [[AVGPOOL0:%.+]] = IE.AvgPool([[CAST0_PER_TENSOR]])
    // CHECK-SAME{LITERAL}: {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME{LITERAL}:  rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME: tensor<16x3x3x3x[[I8_PER_TENSOR]]> -> tensor<16x3x3x3x[[U8_PER_TENSOR]]>
    // CHECK:   [[AVGPOOL0_NORMALIZE:%.+]] = IE.QuantizeCast([[AVGPOOL0]]) {dstElemType = ui8}
    // CHECK:   [[AVGPOOL0_PER_AXIS:%.+]] = IE.QuantizeCast([[AVGPOOL0_NORMALIZE]]) {dstElemType = [[QTYPE2]]}
    // CHECK:   [[REORDER0:%.+]] = IE.Reorder([[AVGPOOL0_PER_AXIS]]) {dstOrder = #NHWC}

    // CHECK:   [[BOUNDARY_CAST0:%.+]] = IE.QuantizeCast([[REORDER0]]) {dstElemType = ui8}

    // CHECK:   [[RESHAPE0:%.+]] = IE.Reshape([[ARG1]]) {shape_value = [1, 10, 1, 20]}
    // CHECK:   [[CAST1:%.+]] = IE.QuantizeCast([[RESHAPE0]]) {dstElemType = [[QTYPE5]]}
    // CHECK:   [[AFFINERESHAPE0:%.+]] = IE.AffineReshape([[CAST1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [10, 20, 1, 1]}
    // CHECK-SAME: tensor<1x10x1x20x[[QTYPE5]]> -> tensor<10x20x1x1x[[QTYPE6]]>
    // CHECK:   [[CAST1_NORMALIZE:%.+]] = IE.QuantizeCast([[AFFINERESHAPE0]]) {dstElemType = si8}
    // CHECK:   [[CAST1_PER_TENSOR:%.+]] = IE.QuantizeCast([[CAST1_NORMALIZE]]) {dstElemType = [[I8_PER_TENSOR]]}
    // CHECK:   [[AVGPOOL1:%.+]] = IE.AvgPool([[CAST1_PER_TENSOR]])
    // CHECK-SAME: {exclude_pads, kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME: rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK-SAME: tensor<10x20x1x1x[[I8_PER_TENSOR]]> -> tensor<10x20x1x1x[[U8_PER_TENSOR]]>
    // CHECK:   [[AVGPOOL1_NORMALIZE:%.+]] = IE.QuantizeCast([[AVGPOOL1]]) {dstElemType = ui8}
    // CHECK:   [[AVGPOOL1_PER_AXIS:%.+]] = IE.QuantizeCast([[AVGPOOL1_NORMALIZE]]) {dstElemType = [[QTYPE7]]}

    // CHECK:   [[BOUNDARY_CAST1:%.+]] = IE.QuantizeCast([[AVGPOOL1_PER_AXIS]]) {dstElemType = ui8}

    // CHECK:   return [[BOUNDARY_CAST0]], [[BOUNDARY_CAST1]]

    // CHECK: func.func private @main
    // CHECK-SAME: ([[INIT_OUT0:%.+]]: tensor<16x3x3x3xui8, {order = #NHWC}>, [[INIT_OUT1:%.+]]: tensor<10x20x1x1xui8>)
    // CHECK:   [[QUANTIZECAST10:%.+]] = VPU.QuantizeCast([[INIT_OUT0]]) {dstElemType = [[QTYPE2]]}
    // CHECK:   [[SLICE0:%.+]] = VPU.Slice [[QUANTIZECAST10]] [0, 0, 0, 0] [16, 1, 3, 3]
    // CHECK:   [[SLICE1:%.+]] = VPU.Slice [[QUANTIZECAST10]] [0, 1, 0, 0] [16, 2, 3, 3]
    // CHECK:   [[QUANTIZECAST12:%.+]] = VPU.QuantizeCast([[INIT_OUT1]]) {dstElemType = [[QTYPE7]]}

    // CHECK:   [[QUANTIZECAST13:%.+]] = VPU.QuantizeCast([[SLICE0]]) {dstElemType = ui8}
    // CHECK:   [[QUANTIZECAST14:%.+]] = VPU.QuantizeCast([[SLICE1]]) {dstElemType = ui8}
    // CHECK:   [[QUANTIZECAST15:%.+]] = VPU.QuantizeCast([[QUANTIZECAST12]]) {dstElemType = ui8}
    // CHECK:   return [[QUANTIZECAST13]], [[QUANTIZECAST14]], [[QUANTIZECAST15]]

    // CHECK: func.func @wrapper_main() -> (tensor<16x1x3x3xui8, {order = #NHWC}>, tensor<16x2x3x3xui8, {order = #NHWC}>, tensor<10x20x1x1xui8>)
    // CHECK-DAG: [[CST0:%.+]] = const.Declare tensor<16x3x3x3xsi8> = dense_resource<ov_0>
    // CHECK-DAG: [[CST1:%.+]] = const.Declare tensor<10x20xsi8> = dense_resource<ov_1>
    // CHECK:   [[CALL:%[0-9]+]]:2 = call @init([[CST0]], [[CST1]])
    // CHECK:   [[RET:%.+]]:3 = call @main([[CALL]]#0, [[CALL]]#1)
    // CHECK:   return [[RET]]#0, [[RET]]#1, [[RET]]#2
}

// -----

{-#
  dialect_resources: {
    builtin: {
            ov_0: "0x10000000AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30AEB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E300EB00E30",
            ov_1: "0x100000000AB0CE30"
        }
  }
#-}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @Convolution
module @Convolution {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x3x62x62xf16>
    } outputsInfo : {
        DataInfo "output_0" : tensor<1x16x60x60xf16>
        DataInfo "output_1" : tensor<2x1x1x1xf16, {order = #NHWC}>
        DataInfo "output_2" : tensor<1x2x1x1xf16>
    }

    // CHECK: IE.CNNNetwork entryPoint : @wrapper_main inputsInfo : {
    // CHECK:     DataInfo "input" : tensor<1x3x62x62xf16>
    // CHECK: } outputsInfo : {
    // CHECK:     DataInfo "output_0" : tensor<1x16x60x60xf16>
    // CHECK:     DataInfo "output_1" : tensor<2x1x1x1xf16, {order = #NHWC}>
    // CHECK:     DataInfo "output_2" : tensor<1x2x1x1xf16>

    func.func @main(%arg0: tensor<1x3x62x62xf16>) -> (tensor<1x16x60x60xf16>, tensor<2x1x1x1xf16, {order = #NHWC}>,  tensor<1x2x1x1xf16>) {
        %cst = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
        %cst_0 = const.Declare tensor<16x16x3x3xf16, {order = #NHWC}>
                      = dense_resource<ov_0> : tensor<16x3x3x3xf32>, [#const.CastElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [0, 13, 0, 0]>]

        %0 = VPU.Expand(%arg0) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
        %1 = VPU.NCE.Permute(%0) {dstElemType = f16, dstOrder = #NHWC, expandedChannels = 16 : i64, ppe = #VPU.PPEStub<>} -> tensor<1x16x62x64xf16, {order = #NHWC}>
        %2 = VPU.Slice %1 [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
        %3 = VPU.NCE.Convolution(%2, %cst_0, %cst) {
              pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>, rawFilterShape = [16, 16, 3, 3], strides = [1, 1], ppe = #VPU.PPEStub<>}
                  -> tensor<1x16x60x60xf16>

        %cst_1 = const.Declare tensor<2x1x1x1xf16, {order = #NHWC}> = dense_resource<ov_1> : tensor<1x2x1x1xf16>, [#const.Reshape<[2, 1, 1, 1]>, #const.Reorder<#NHWC>]
        %cst_2 = const.Declare tensor<1x2x1x1xf16> = dense_resource<ov_1> : tensor<1x2x1x1xf16>, [#const.Add<1.0>]

        return %3, %cst_1, %cst_2 : tensor<1x16x60x60xf16>, tensor<2x1x1x1xf16, {order = #NHWC}>,  tensor<1x2x1x1xf16>
    }

    // CHECK:       func.func private @init([[OV_CONST0:%.+]]: tensor<16x3x3x3xf32>, [[OV_CONST1:%.+]]: tensor<1x2x1x1xf16>)
    // CHECK:           [[CONVERT0:%.+]] = IE.Convert([[OV_CONST0]]) {dstElemType = f16} : tensor<16x3x3x3xf32> -> tensor<16x3x3x3xf16>
    // CHECK:           [[REORDER0:%.+]] = IE.Reorder([[CONVERT0]]) {dstOrder = #NHWC} : tensor<16x3x3x3xf16> -> tensor<16x3x3x3xf16, {order = #NHWC}>
    // CHECK:           [[PAD0:%.+]] = IE.Pad([[REORDER0]]) {mode = #IE.pad_mode<CONSTANT>, pad_value_attr = 0.000000e+00 : f64,
    // CHECK-SAME:                      pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [0, 13, 0, 0]} : tensor<16x3x3x3xf16, {order = #NHWC}> -> tensor<16x16x3x3xf16, {order = #NHWC}>
    // CHECK:           [[CST:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf32>, [#const.CastElemType<f16>]
    // CHECK:           [[ADD0:%.+]] = IE.Add([[OV_CONST1]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x2x1x1xf16>, tensor<1xf16> -> tensor<1x2x1x1xf16>
    // CHECK:           return [[PAD0]], [[ADD0]]


    // CHECK:       func.func private @main([[ARG0:%.+]]: tensor<1x3x62x62xf16>, [[INIT_OUT0:%.+]]: tensor<16x16x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:                      [[INIT_OUT2:%.+]]: tensor<1x2x1x1xf16>)
    // CHECK:           [[CST:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    // CHECK:           [[EXPAND0:%.+]] = VPU.Expand([[ARG0]]) {pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 2]} : tensor<1x3x62x62xf16> -> tensor<1x3x62x64xf16>
    // CHECK:           [[PERMUTE0:%.+]] = VPU.NCE.Permute([[EXPAND0]])
    // CHECK:           [[SLICE0:%.+]] = VPU.Slice [[PERMUTE0]] [0, 0, 0, 0] [1, 16, 62, 62] : tensor<1x16x62x64xf16, {order = #NHWC}> to tensor<1x16x62x62xf16, {order = #NHWC}>
    // CHECK:           [[CONVOLUTION0:%.+]] = VPU.NCE.Convolution([[SLICE0]], [[INIT_OUT0]], [[CST]])
    // CHECK:           [[CST2:%.+]] = const.Declare tensor<2x1x1x1xf16, {order = #NHWC}> = dense_resource<ov_1>
    // CHECK-SAME:          [#const.Reshape<[2, 1, 1, 1]>, #const.Reorder<#NHWC>]
    // CHECK:           return [[CONVOLUTION0]], [[CST2]], [[INIT_OUT2]]

    // CHECK:       func.func @wrapper_main([[ARG0:%.+]]: tensor<1x3x62x62xf16>) -> (tensor<1x16x60x60xf16>, tensor<2x1x1x1xf16, {order = #NHWC}>, tensor<1x2x1x1xf16>)
    // CHECK-DAG:       [[CST0:%.+]] = const.Declare tensor<16x3x3x3xf32> = dense_resource<ov_0> : tensor<16x3x3x3xf32>
    // CHECK-DAG:       [[CST1:%.+]] = const.Declare tensor<1x2x1x1xf16> = dense_resource<ov_1> : tensor<1x2x1x1xf16>
    // CHECK:           [[CALL:%[0-9]+]]:2 = call @init([[CST0]], [[CST1]])
    // CHECK:           [[RET:%.+]]:3 = call @main([[ARG0]], [[CALL]]#0, [[CALL]]#1)
    // CHECK:           return [[RET]]#0, [[RET]]#1, [[RET]]#2
}

// -----

{-#
  dialect_resources: {
    builtin: {
            ov_0: "0x10000000ABCDABCDABCDABCE"
        }
  }
#-}

!qElemType = !quant.uniform<i8:f16:1, {8.9925130208333328E-4, 5.9925130208333328E-4}>

// CHECK-LABEL: @QuantizeAttr
module @QuantizeAttr {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<2x2xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf16>
    }

    // CHECK: IE.CNNNetwork entryPoint : @wrapper_main inputsInfo : {
    // CHECK:     DataInfo "input" : tensor<2x2xf16>
    // CHECK: } outputsInfo : {
    // CHECK:     DataInfo "output" : tensor<2x2xf16>

    func.func @main(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst = const.Declare tensor<2x2x!qElemType> = dense_resource<ov_0> : tensor<2x2xf16>, [#const.Quantize<!qElemType>]
        return %dummy : tensor<2x2xf16>
    }

    // CHECK:       func.func private @init([[OV_CONST0:%.+]]: tensor<2x2xf16>)
    // CHECK:           [[QUANTIZE:%.+]] = IE.Quantize([[OV_CONST0]]) {dstElemType = !qElemType} : tensor<2x2xf16> -> tensor<2x2x!qElemType>
    // CHECK:           [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[QUANTIZE]]) {dstElemType = si8} : tensor<2x2x!qElemType> -> tensor<2x2xsi8>
    // CHECK:           return [[QUANTIZE_CAST]] : tensor<2x2xsi8>

    // CHECK:       func.func private @main([[ARG0:%.+]]: tensor<2x2xf16>, [[ARG1:%.+]]: tensor<2x2xsi8>) -> tensor<2x2xf16>
    // CHECK:           [[QUANTIZE_CAST_1:%.+]] = VPU.QuantizeCast([[ARG1]]) {dstElemType = !qElemType} : tensor<2x2xsi8> -> tensor<2x2x!qElemType>
    // CHECK:           return [[ARG0]] : tensor<2x2xf16>

    // CHECK:       func.func @wrapper_main([[ARG2:%.+]]: tensor<2x2xf16>) -> tensor<2x2xf16>
    // CHECK:           [[CST:%.+]] = const.Declare tensor<2x2xf16> = dense_resource<ov_0> : tensor<2x2xf16>
    // CHECK:           [[CALL0:%.+]] = call @init([[CST]]) : (tensor<2x2xf16>) -> tensor<2x2xsi8>
    // CHECK:           [[CALL1:%.+]] = call @main([[ARG2]], [[CALL0]]) : (tensor<2x2xf16>, tensor<2x2xsi8>) -> tensor<2x2xf16>
    // CHECK:           return [[CALL1]] : tensor<2x2xf16>
}

// -----

{-#
  dialect_resources: {
    builtin: {
            ov_0: "0x10000000ABCDABCDABCDABCE"
        }
  }
#-}

// Test that same base content, same type constants are not fused together when
// transformations differ marginally.

// CHECK-LABEL: @UniqueArgumentChains
module @UniqueArgumentChains {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<2x2xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf16>
    }

    // CHECK: IE.CNNNetwork entryPoint : @wrapper_main inputsInfo : {
    // CHECK:     DataInfo "input" : tensor<2x2xf16>
    // CHECK: } outputsInfo : {
    // CHECK:     DataInfo "output" : tensor<2x2xf16>

    func.func @main(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst0 = const.Declare tensor<2x2xf16> = dense_resource<ov_0> : tensor<2x2xf16>, [#const.Add<1.0>]
        %cst1 = const.Declare tensor<2x2xf16> = dense_resource<ov_0> : tensor<2x2xf16>, [#const.Add<2.0>]
        return %dummy : tensor<2x2xf16>
    }

    // CHECK:       func.func private @init([[OV_CONST0:%.+]]: tensor<2x2xf16>) -> (tensor<2x2xf16>, tensor<2x2xf16>)
    // CHECK:           [[CST2:%.+]] = const.Declare {{.*}} dense<2.000000e+00>
    // CHECK:           [[ADD2:%.+]] = IE.Add([[OV_CONST0]], [[CST2]])
    // CHECK:           [[CST1:%.+]] = const.Declare {{.*}} dense<1.000000e+00>
    // CHECK:           [[ADD1:%.+]] = IE.Add([[OV_CONST0]], [[CST1]])
    // CHECK:           return [[ADD2]], [[ADD1]]

    // CHECK:       func.func private @main([[ARG0:%.+]]: tensor<2x2xf16>, [[INIT0:%.+]]: tensor<2x2xf16>, [[INIT1:%.+]]: tensor<2x2xf16>)
    // CHECK:           return [[ARG0]]

    // CHECK:       func.func @wrapper_main([[ARG2:%.+]]: tensor<2x2xf16>) -> tensor<2x2xf16>
    // CHECK:           [[CST:%.+]] = const.Declare tensor<2x2xf16> = dense_resource<ov_0> : tensor<2x2xf16>
    // CHECK:           [[INIT:%.+]]:2 = call @init([[CST]])
    // CHECK:           [[MAIN:%.+]] = call @main([[ARG2]], [[INIT]]#0, [[INIT]]#1)
    // CHECK:           return [[MAIN]]
}

// -----

{-#
  dialect_resources: {
    builtin: {
            ov_0: "0x10000000ABCDABCDABCDABCE"
        }
  }
#-}

// CHECK-LABEL: @OutlinedConstants
module @OutlinedConstants {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<2x2xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf16>
    }

    // CHECK: IE.CNNNetwork entryPoint : @wrapper_main inputsInfo : {
    // CHECK:     DataInfo "input" : tensor<2x2xf16>
    // CHECK: } outputsInfo : {
    // CHECK:     DataInfo "output" : tensor<2x2xf16>

    func.func private @main_foo1(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst = const.Declare tensor<2x2xf16> = dense_resource<ov_0> : tensor<2x2xf16>, [#const.Add<15.0>]
        %cst_bar_duplicate = const.Declare tensor<2x2xf16> = dense_resource<ov_0> : tensor<2x2xf16>,
            [#const.Rescale<2.0>]
        %user_cst = VPU.Convert(%cst) {dstElemType = f32} : tensor<2x2xf16> -> tensor<2x2xf32>
        %user_cst_bar_duplicate = VPU.Convert(%cst_bar_duplicate) {dstElemType = f32}
            : tensor<2x2xf16> -> tensor<2x2xf32>
        return %dummy : tensor<2x2xf16>
    }

    // CHECK:   func.func private @main_foo1([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST:%.+]]: tensor<2x2xf16>, [[CST_BAR_DUPLICATE:%.+]]: tensor<2x2xf16>)
    // CHECK:       [[USER_CST:%.+]] = VPU.Convert([[CST]]) {dstElemType = f32}
    // CHECK:       [[USER_CST_BAR_DUPLICATE:%.+]] = VPU.Convert([[CST_BAR_DUPLICATE]]) {dstElemType = f32}
    // CHECK:       return [[DUMMY]]

    func.func private @main_bar() -> (tensor<4x1xf16>, tensor<2x2xf16>) {
        %cst1 = const.Declare tensor<4x1xf16> = dense_resource<ov_0> : tensor<2x2xf16>, [#const.Add<15.0>, #const.Reshape<[4, 1]>]
        %cst2 = const.Declare tensor<2x2xf16> = dense_resource<ov_0> : tensor<2x2xf16>, [#const.Rescale<2.0>]
        return %cst1, %cst2 : tensor<4x1xf16>, tensor<2x2xf16>
    }

    // CHECK:   func.func private @main_bar([[CST1:%.+]]: tensor<4x1xf16>, [[CST2:%.+]]: tensor<2x2xf16>)
    // CHECK:       return [[CST1]], [[CST2]]

    func.func private @main_foo2(%dummy: tensor<2x2xf16>) -> (tensor<2x2xf16>, tensor<4x1xf16>, tensor<2x2xf16>) {
        %cst = const.Declare tensor<2x2xf16> = dense_resource<ov_0> : tensor<2x2xf16>, [#const.Add<10.0>]
        %cst_bar_duplicate = const.Declare tensor<2x2xf16> = dense_resource<ov_0> : tensor<2x2xf16>,
            [#const.Rescale<2.0>]

        %user_cst = VPU.Convert(%cst) {dstElemType = f32} : tensor<2x2xf16> -> tensor<2x2xf32>
        %user_cst_bar_duplicate = VPU.Convert(%cst_bar_duplicate) {dstElemType = f32}
            : tensor<2x2xf16> -> tensor<2x2xf32>

        %call:2 = func.call @main_bar() : () -> (tensor<4x1xf16>, tensor<2x2xf16>)
        return %dummy, %call#0, %call#1 : tensor<2x2xf16>, tensor<4x1xf16>, tensor<2x2xf16>
    }

    // CHECK:   func.func private @main_foo2([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST_BAR_DUPLICATE:%.+]]: tensor<2x2xf16>, [[CST:%.+]]: tensor<2x2xf16>, [[BAR_CST1:%.+]]: tensor<4x1xf16>)
    // CHECK:       [[USER_CST:%.+]] = VPU.Convert([[CST]]) {dstElemType = f32}
    // CHECK:       [[USER_CST_BAR_DUPLICATE:%.+]] = VPU.Convert([[CST_BAR_DUPLICATE]]) {dstElemType = f32}
    // CHECK:       [[CALL:%.+]]:2 = call @main_bar([[BAR_CST1]], [[CST_BAR_DUPLICATE]])
    // CHECK:       return [[DUMMY]], [[CALL]]#0, [[CALL]]#1


    // CHECK:   func.func private @init([[OV_CONST0:%.+]]: tensor<2x2xf16>)
    // CHECK-SAME:  -> (tensor<2x2xf16>, tensor<2x2xf16>, tensor<2x2xf16>, tensor<4x1xf16>)

    // foo2 && main: dense_resource<ov_0> : tensor<2x2xf16>, [#const.Add<10.0>]

    // CHECK:       [[CST3:%.+]] = const.Declare {{.*}} dense<1.000000e+01>
    // CHECK:       [[CST_ADD10:%.+]] = IE.Add([[OV_CONST0]], [[CST3]])

    // foo1: dense_resource<ov_0> : tensor<2x2xf16>, [#const.Add<15.0>]
    // foo2: dense_resource<ov_0> : tensor<2x2xf16>, [#const.Add<15.0>]

    // CHECK:       [[CST1:%.+]] = const.Declare {{.*}} dense<1.500000e+01>
    // CHECK:       [[CST_ADD15:%.+]] = IE.Add([[OV_CONST0]], [[CST1]])

    // foo2 && bar:  dense_resource<ov_0> : tensor<2x2xf16>, [#const.Rescale<2.0>]

    // CHECK:       [[CST2:%.+]] = const.Declare {{.*}} dense<2.000000e+00>
    // CHECK:       [[CST_RESCALE2:%.+]] = IE.Multiply([[OV_CONST0]], [[CST2]])

    // bar:  dense_resource<ov_0> : tensor<2x2xf16>, [#const.Add<15.0>, #const.Reshape<[4, 1]>]

    // CHECK:       [[CST_RESHAPE_4_1:%.+]] = IE.Reshape([[CST_ADD15]]) {shape_value = [4, 1]}

    // CHECK:       return [[CST_ADD10]], [[CST_ADD15]], [[CST_RESCALE2]], [[CST_RESHAPE_4_1]]


    func.func @main(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst0 = const.Declare tensor<2x2xf16> = dense_resource<ov_0> : tensor<2x2xf16>, [#const.Add<10.0>]
        %user_cst0 = VPU.Convert(%cst0) {dstElemType = f32} : tensor<2x2xf16> -> tensor<2x2xf32>

        %call_foo1 = func.call @main_foo1(%dummy) : (tensor<2x2xf16>) -> tensor<2x2xf16>
        %call_foo2:3 = func.call @main_foo2(%dummy): (tensor<2x2xf16>) -> (tensor<2x2xf16>, tensor<4x1xf16>, tensor<2x2xf16>)

        %user_foo1 = VPU.Convert(%call_foo1) {dstElemType = f32} : tensor<2x2xf16> -> tensor<2x2xf32>

        %user_foo2_0 = VPU.Convert(%call_foo2#0) {dstElemType = f32} : tensor<2x2xf16> -> tensor<2x2xf32>
        %user_foo2_1 = VPU.Convert(%call_foo2#1) {dstElemType = f32} : tensor<4x1xf16> -> tensor<4x1xf32>
        %user_foo2_2 = VPU.Convert(%call_foo2#2) {dstElemType = f32} : tensor<2x2xf16> -> tensor<2x2xf32>

        return %dummy : tensor<2x2xf16>
    }

    // CHECK:   func.func private @main([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST_ADD10:%.+]]: tensor<2x2xf16>, [[CST_ADD15:%.+]]: tensor<2x2xf16>, [[CST_RESCALE2:%.+]]: tensor<2x2xf16>, [[CST_RESHAPE_4_1:%.+]]: tensor<4x1xf16>)
    // CHECK:       [[USER_CST0:%.+]] = VPU.Convert([[CST_ADD10]]) {dstElemType = f32}
    // CHECK:       [[CALL_FOO1:%.+]] = call @main_foo1([[DUMMY]], [[CST_ADD15]], [[CST_RESCALE2]])
    // CHECK:       [[CALL_FOO2:%.+]]:3 = call @main_foo2([[DUMMY]], [[CST_RESCALE2]], [[CST_ADD10]], [[CST_RESHAPE_4_1]])
    // CHECK:       [[USER_CALL_FOO1:%.+]] = VPU.Convert([[CALL_FOO1]]) {dstElemType = f32}
    // CHECK:       [[USER_CALL_FOO2_0:%.+]] = VPU.Convert([[CALL_FOO2]]#0) {dstElemType = f32}
    // CHECK:       [[USER_CALL_FOO2_1:%.+]] = VPU.Convert([[CALL_FOO2]]#1) {dstElemType = f32}
    // CHECK:       [[USER_CALL_FOO2_2:%.+]] = VPU.Convert([[CALL_FOO2]]#2) {dstElemType = f32}
    // CHECK:       return [[DUMMY]]


    // CHECK:   func.func @wrapper_main([[DUMMY:%.+]]: tensor<2x2xf16>)
    // CHECK:       [[OV_0:%.+]] = const.Declare {{.*}} dense_resource<ov_0>
    // CHECK:       [[INIT:%.+]]:4 = call @init([[OV_0]])
    // CHECK:       [[MAIN:%.+]] = call @main([[DUMMY]], [[INIT]]#0, [[INIT]]#1, [[INIT]]#2, [[INIT]]#3)
    // CHECK:       return [[MAIN]]
}

// -----

{-#
  dialect_resources: {
    builtin: {
            ov_0: "0x10000000ABCDABCDABCDABCE",
            ov_1: "0x10000000ABCDABCDABCDABCE"
        }
  }
#-}

// CHECK-LABEL: @OutlinedConstants_MultiCall
module @OutlinedConstants_MultiCall {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<2x2xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf16>
    }

    // CHECK: IE.CNNNetwork entryPoint : @wrapper_main inputsInfo : {
    // CHECK:     DataInfo "input" : tensor<2x2xf16>
    // CHECK: } outputsInfo : {
    // CHECK:     DataInfo "output" : tensor<2x2xf16>

    func.func private @multi_call(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst = const.Declare tensor<2x2xf16> = dense_resource<ov_0> : tensor<2x2xf16>, [#const.Rescale<42.0>]
        %user_cst = VPU.Convert(%cst) {dstElemType = f32} : tensor<2x2xf16> -> tensor<2x2xf32>
        return %dummy : tensor<2x2xf16>
    }

    // CHECK:   func.func private @multi_call([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST:%.+]]: tensor<2x2xf16>)
    // CHECK:       [[USER_CST:%.+]] = VPU.Convert([[CST]]) {dstElemType = f32}
    // CHECK:       return [[DUMMY]]

    func.func private @single_call(%dummy: tensor<2x2xf16>) -> (tensor<2x2xf16>, tensor<2x2xf16>) {
        %cst1 = const.Declare tensor<2x2xf16> = dense_resource<ov_1> : tensor<2x2xf16>, [#const.Add<15.0>]
        %call = func.call @multi_call(%dummy) : (tensor<2x2xf16>) -> tensor<2x2xf16>
        return %cst1, %call : tensor<2x2xf16>, tensor<2x2xf16>
    }

    // CHECK:   func.func private @single_call([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST1:%.+]]: tensor<2x2xf16>, [[MULTI_CALL_CST:%.+]]: tensor<2x2xf16>)
    // CHECK:       [[CALL:%.+]] = call @multi_call([[DUMMY]], [[MULTI_CALL_CST]])
    // CHECK:       return [[CST1]], [[CALL]]


    // CHECK:   func.func private @init([[OV_CONST0:%.+]]: tensor<2x2xf16>, [[OV_CONST1:%.+]]: tensor<2x2xf16>)
    // CHECK-SAME:  -> (tensor<2x2xf16>, tensor<2x2xf16>)
    // CHECK:       [[CST1:%.+]] = const.Declare {{.*}} dense<4.200000e+01>
    // CHECK:       [[CST_RESCALE_42:%.+]] = IE.Multiply([[OV_CONST0]], [[CST1]])
    // CHECK:       [[CST2:%.+]] = const.Declare {{.*}} dense<1.500000e+01>
    // CHECK:       [[CST_ADD15:%.+]] = IE.Add([[OV_CONST1]], [[CST2]])
    // CHECK:       return [[CST_RESCALE_42]], [[CST_ADD15]]


    func.func @main(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        // -> multi_call
        %call_multi1 = func.call @multi_call(%dummy) : (tensor<2x2xf16>) -> tensor<2x2xf16>
        // -> single_call -> multi_call
        %call_single:2 = func.call @single_call(%dummy) : (tensor<2x2xf16>) -> (tensor<2x2xf16>, tensor<2x2xf16>)
        // -> multi_call (again)
        %call_multi2 = func.call @multi_call(%dummy) : (tensor<2x2xf16>) -> tensor<2x2xf16>
        return %dummy : tensor<2x2xf16>
    }

    // CHECK:   func.func private @main([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST_RESCALE_42:%.+]]: tensor<2x2xf16>, [[CST_ADD_15:%.+]]: tensor<2x2xf16>)
    // CHECK:       [[CALL_MULTI1:%.+]] = call @multi_call([[DUMMY]], [[CST_RESCALE_42]])
    // CHECK:       [[CALL_SINGLE:%.+]]:2 = call @single_call([[DUMMY]], [[CST_ADD_15]], [[CST_RESCALE_42]])
    // CHECK:       [[CALL_MULTI2:%.+]] = call @multi_call([[DUMMY]], [[CST_RESCALE_42]])
    // CHECK:       return [[DUMMY]]


    // CHECK:   func.func @wrapper_main([[DUMMY:%.+]]: tensor<2x2xf16>)
    // CHECK:       [[OV_0:%.+]] = const.Declare {{.*}} dense_resource<ov_0>
    // CHECK:       [[OV_1:%.+]] = const.Declare {{.*}} dense_resource<ov_1>
    // CHECK:       [[INIT:%.+]]:2 = call @init([[OV_0]], [[OV_1]])
    // CHECK:       [[MAIN:%.+]] = call @main([[DUMMY]], [[INIT]]#0, [[INIT]]#1)
    // CHECK:       return [[MAIN]]
}


// -----

!qElemType1 = !quant.uniform<i8:f16, 0.5>
// CHECK-DAG: [[QTYPE1:!.+]] = !quant.uniform<i8:f16, 5.000000e-01>
!qElemType2 = !quant.uniform<u8:f16, 0.5>
// CHECK-DAG: [[QTYPE2:!.+]] = !quant.uniform<u8:f16, 5.000000e-01>

{-#
  dialect_resources: {
    builtin: {
            ov_0: "0x10000000ABCDABCDABCDABCE"
        }
  }
#-}

// This tests how I/O boundaries are handled when dealing with outlining,
// especially when the same constant is used in both the caller and the callee.

// CHECK: @OutlinedConstants_Quantized
module @OutlinedConstants_Quantized {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<2x2xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf16>
    }

    // CHECK: IE.CNNNetwork entryPoint : @wrapper_main inputsInfo : {
    // CHECK:     DataInfo "input" : tensor<2x2xf16>
    // CHECK: } outputsInfo : {
    // CHECK:     DataInfo "output" : tensor<2x2xf16>

    func.func private @quant_cst(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst = const.Declare tensor<2x2x!qElemType1> = dense_resource<ov_0> : tensor<2x2xf16>,
            [#const.CastElemType<!qElemType1>]
        return %dummy : tensor<2x2xf16>
    }

    // CHECK:   func.func private @quant_cst([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST:%.+]]: tensor<2x2xsi8>)
    // CHECK:       [[CAST:%.+]] = VPU.QuantizeCast([[CST]]) {dstElemType = [[QTYPE1]]}
    // CHECK:       return [[DUMMY]]


    // CHECK:   func.func private @init([[OV_CONST0:%.+]]: tensor<2x2xf16>)
    // CHECK-SAME:  -> (tensor<2x2xui8>, tensor<2x2xsi8>)
    // CHECK:       [[CVT_U8:%.+]] = IE.Convert([[OV_CONST0]]) {dstElemType = i8}
    // CHECK:       [[CST_QTYPE2:%.+]] = IE.QuantizeCast([[CVT_U8]]) {dstElemType = [[QTYPE2]]}
    // CHECK:       [[CST_QTYPE2_FIXED:%.+]] = IE.QuantizeCast([[CST_QTYPE2]]) {dstElemType = ui8}

    // CHECK:       [[CVT_I8:%.+]] = IE.Convert([[OV_CONST0]]) {dstElemType = i8}
    // CHECK:       [[CST_QTYPE1:%.+]] = IE.QuantizeCast([[CVT_I8]]) {dstElemType = [[QTYPE1]]}
    // CHECK:       [[CST_QTYPE1_FIXED:%.+]] = IE.QuantizeCast([[CST_QTYPE1]]) {dstElemType = si8}

    // CHECK:       return [[CST_QTYPE2_FIXED]], [[CST_QTYPE1_FIXED]]


    func.func @main(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst = const.Declare tensor<2x2x!qElemType1> = dense_resource<ov_0> : tensor<2x2xf16>,
            [#const.CastElemType<!qElemType1>]
        %cst2 = const.Declare tensor<2x2x!qElemType2> = dense_resource<ov_0> : tensor<2x2xf16>,
            [#const.CastElemType<!qElemType2>]
        %call = func.call @quant_cst(%dummy) : (tensor<2x2xf16>) -> tensor<2x2xf16>
        return %call : tensor<2x2xf16>
    }

    // CHECK:   func.func private @main([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST_QTYPE2_BAD:%.+]]: tensor<2x2xui8>, [[CST_QTYPE1_BAD:%.+]]: tensor<2x2xsi8>)
    // CHECK:       [[CST_QTYPE2_GOOD:%.+]] = VPU.QuantizeCast([[CST_QTYPE2_BAD]]) {dstElemType = [[QTYPE2]]}
    // CHECK:       [[CST_QTYPE1_GOOD:%.+]] = VPU.QuantizeCast([[CST_QTYPE1_BAD]]) {dstElemType = [[QTYPE1]]}
    // CHECK:       [[CALL:%.+]] = call @quant_cst([[DUMMY]], [[CST_QTYPE1_BAD]])
    // CHECK:       return [[CALL]]


    // CHECK:   func.func @wrapper_main([[DUMMY:%.+]]: tensor<2x2xf16>)
    // CHECK:       [[OV_0:%.+]] = const.Declare {{.*}} dense_resource<ov_0>
    // CHECK:       [[INIT:%.+]]:2 = call @init([[OV_0]])
    // CHECK:       [[MAIN:%.+]] = call @main([[DUMMY]], [[INIT]]#0, [[INIT]]#1)
    // CHECK:       return [[MAIN]]
}


// -----

{-#
  dialect_resources: {
    builtin: {
            ov_0: "0x10000000ABCDABCDABCDABCE"
        }
  }
#-}

// This tests how post-init transformations are generated when dealing with
// outlining, especially when same post-init transformation is present in both
// the caller and the callee.

// CHECK-LABEL: @OutlinedConstants_PostInitTransformations
module @OutlinedConstants_PostInitTransformations {
    IE.CNNNetwork entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<2x2xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf16>
    }

    // CHECK: IE.CNNNetwork entryPoint : @wrapper_main inputsInfo : {
    // CHECK:     DataInfo "input" : tensor<2x2xf16>
    // CHECK: } outputsInfo : {
    // CHECK:     DataInfo "output" : tensor<2x2xf16>

    func.func private @subview_cst(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst = const.Declare tensor<2x1xf16> = dense_resource<ov_0> : tensor<2x2xf16>,
            [#const.Add<42.0>, #const.SubView<[0, 1], [2, 1]>]
        return %dummy : tensor<2x2xf16>
    }

    // CHECK:   func.func private @subview_cst([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST:%.+]]: tensor<2x2xf16>)
    // CHECK:       [[SUBVIEW_2_1:%.+]] = VPU.Slice [[CST]] [0, 1] [2, 1]
    // CHECK:       return [[DUMMY]]


    // CHECK:   func.func private @init([[OV_CONST0:%.+]]: tensor<2x2xf16>) -> tensor<2x2xf16
    // CHECK:       [[CST:%.+]] = const.Declare {{.*}} dense<4.200000e+01>
    // CHECK:       [[CST_ADD42:%.+]] = IE.Add([[OV_CONST0]], [[CST]])
    // CHECK:       return [[CST_ADD42]]


    func.func @main(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %cst = const.Declare tensor<2x1xf16> = dense_resource<ov_0> : tensor<2x2xf16>,
            [#const.Add<42.0>, #const.SubView<[0, 1], [2, 1]>]
        %cst2 = const.Declare tensor<1x1xf16> = dense_resource<ov_0> : tensor<2x2xf16>,
            [#const.Add<42.0>, #const.SubView<[0, 0], [1, 1]>]
        %call = func.call @subview_cst(%dummy) : (tensor<2x2xf16>) -> tensor<2x2xf16>
        return %call : tensor<2x2xf16>
    }

    // CHECK:   func.func private @main([[DUMMY:%.+]]: tensor<2x2xf16>, [[CST_ADD42:%.+]]: tensor<2x2xf16>)
    // CHECK:       [[SUBVIEW_2_1:%.+]] = VPU.Slice [[CST_ADD42]] [0, 1] [2, 1]
    // CHECK:       [[SUBVIEW_1_1:%.+]] = VPU.Slice [[CST_ADD42]] [0, 0] [1, 1]
    // CHECK:       [[CALL:%.+]] = call @subview_cst([[DUMMY]], [[CST_ADD42]])
    // CHECK:       return [[CALL]]


    // CHECK:   func.func @wrapper_main([[DUMMY:%.+]]: tensor<2x2xf16>)
    // CHECK:       [[OV_0:%.+]] = const.Declare {{.*}} dense_resource<ov_0>
    // CHECK:       [[INIT:%.+]] = call @init([[OV_0]])
    // CHECK:       [[MAIN:%.+]] = call @main([[DUMMY]], [[INIT]])
    // CHECK:       return [[MAIN]]
}
