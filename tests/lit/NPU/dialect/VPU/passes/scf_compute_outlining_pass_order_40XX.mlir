//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This test verifies that FinalizeComputeFunctionBoundaries must run before PackNestedModules
// when compute functions have quantized output types. FinalizeComputeFunctionBoundaries converts
// quantized types to storage types before PackNestedModules creates DataInfo operations.

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --scf-ops-outlining %s | FileCheck %s
// REQUIRES: platform-NPU4000

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<u8:f16, 0.0019697112195632038>

module @PassOrderingScfComputeOpsOutlining {
    net.NetworkInfo entryPoint : @main
        inputsInfo : {
            DataInfo "input0" tensorNames = ["input0"] : tensor<1x3x64x640xf32>
            DataInfo "input1" tensorNames = ["input1"] : tensor<1x3x64x640xf32>
        }
        outputsInfo : {
            DataInfo "output0" tensorNames = ["output0"] : tensor<1x64x640x16xi8>
            DataInfo "output1" tensorNames = ["output1"] : tensor<1x16x64x640xf32>
        }

    func.func @main(
            %arg0: tensor<1x3x64x640xf32>,
            %arg1: tensor<1x3x64x640xf32, {order = #NHWC}>
        )
        -> (tensor<1x16x64x640x!qElemType, {order = #NHWC}>,
            tensor<1x16x64x640xf32>
         ) {
        %0:2 = call @main_func0_static(%arg0, %arg1) : (tensor<1x3x64x640xf32>, tensor<1x3x64x640xf32, {order = #NHWC}>)
            -> (tensor<1x16x64x640x!qElemType, {order = #NHWC}>, tensor<1x16x64x640xf32>)
        return %0#0, %0#1 : tensor<1x16x64x640x!qElemType, {order = #NHWC}>, tensor<1x16x64x640xf32>
    }

    func.func @main_func0_static(%arg0: tensor<1x3x64x640xf32>, %arg1: tensor<1x3x64x640xf32, {order = #NHWC}>)
            -> (tensor<1x16x64x640x!qElemType, {order = #NHWC}>, tensor<1x16x64x640xf32>) {
        %0 = builtin.unrealized_conversion_cast %arg0 : tensor<1x3x64x640xf32> to tensor<1x16x64x640x!qElemType, {order = #NHWC}>
        %1 = builtin.unrealized_conversion_cast %arg1 : tensor<1x3x64x640xf32, {order = #NHWC}> to tensor<1x16x64x640xf32>
        return %0, %1 : tensor<1x16x64x640x!qElemType, {order = #NHWC}>, tensor<1x16x64x640xf32>
    }
}

// CHECK-LABEL: @PassOrderingScfComputeOpsOutlining

// CHECK: func.func @main_func0_static
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x64x640xf32>, [[ARG1:%.+]]: tensor<1x64x640x3xf32>)
// CHECK-SAME: -> (tensor<1x64x640x16xi8>, tensor<1x16x64x640xf32>)
// CHECK-DAG: [[CAST0:%.+]] = Core.ReinterpretCast([[ARG1]]) : tensor<1x64x640x3xf32> -> tensor<1x3x64x640xf32, {order = #NHWC}>
// CHECK-DAG: [[CAST2:%.+]] = Core.ReinterpretCast([[ARG0]]) : tensor<1x3x64x640xf32> -> tensor<1x3x64x640xf32>
// CHECK-DAG: [[UNREALIZED0:%.+]] = builtin.unrealized_conversion_cast [[CAST2]] : tensor<1x3x64x640xf32> to tensor<1x16x64x640x!qElemType, {order = #NHWC}>
// CHECK-DAG: [[CAST1:%.+]] = Core.ReinterpretCast([[UNREALIZED0]]) : tensor<1x16x64x640x!qElemType, {order = #NHWC}> -> tensor<1x64x640x16xi8>
// CHECK-DAG: [[UNREALIZED1:%.+]] = builtin.unrealized_conversion_cast [[CAST0]] : tensor<1x3x64x640xf32, {order = #NHWC}> to tensor<1x16x64x640xf32>
// CHECK-DAG: [[CAST3:%.+]] = Core.ReinterpretCast([[UNREALIZED1]]) : tensor<1x16x64x640xf32> -> tensor<1x16x64x640xf32>
// CHECK: return [[CAST1]], [[CAST3]] : tensor<1x64x640x16xi8>, tensor<1x16x64x640xf32>

// CHECK: func.func @main
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x3x64x640xf32>, [[ARG1:%.+]]: tensor<1x64x640x3xf32>)
// CHECK-SAME: -> (tensor<1x64x640x16xi8>, tensor<1x16x64x640xf32>)
// CHECK: [[RESULTS:%.+]]:2 = Core.NestedCall @Module0::@main_func0_static([[ARG0]], [[ARG1]])
// CHECK-SAME: : (tensor<1x3x64x640xf32>, tensor<1x64x640x3xf32>) -> (tensor<1x64x640x16xi8>, tensor<1x16x64x640xf32>)
// CHECK: return [[RESULTS]]#0, [[RESULTS]]#1 : tensor<1x64x640x16xi8>, tensor<1x16x64x640xf32>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map_h = affine_map<(d0) -> (d0 - 1, 0)>
#map_w = affine_map<(d0) -> (d0 - 2, 0)>

module @LoopInvariantCodeMotionTest {
    net.NetworkInfo entryPoint : @main
        inputsInfo : {
            DataInfo "input0" tensorNames = ["input0"] : tensor<1x16x128x128xf16>
            DataInfo "input1" tensorNames = ["input1"] : tensor<1x16x128x128xf16>
        }
        outputsInfo : {
            DataInfo "output" tensorNames = ["output"] : tensor<1x16x128x128xf16>
        }

    func.func @main(%arg0: tensor<1x16x128x128xf16, {order = #NHWC}>,
                    %arg1: tensor<1x16x128x128xf16, {order = #NHWC}>)
                    -> tensor<1x16x128x128xf16, {order = #NHWC}> {
        %cst_bias = const.Declare tensor<1x16x32x16xf16, {order = #NHWC}> = dense<0.5> : tensor<1x16x32x16xf16>, [#const.Reorder<#NHWC>]

        %output_buff = tensor.empty() : tensor<1x16x128x128xf16, {order = #NHWC}>

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c128 = arith.constant 128 : index
        %c32 = arith.constant 32 : index
        %c16 = arith.constant 16 : index
        %result_h = scf.for %iv_h = %c0 to %c128 step %c32 iter_args(%iter_buf_h = %output_buff) -> (tensor<1x16x128x128xf16, {order = #NHWC}>) {
            %result_w = scf.for %iv_w = %c0 to %c128 step %c16 iter_args(%iter_buf_w = %iter_buf_h) -> (tensor<1x16x128x128xf16, {order = #NHWC}>) {
                %h_slice_offset = affine.max #map_h(%iv_h)
                %w_slice_offset = affine.max #map_w(%iv_w)

                %input0_slice = tensor.extract_slice %arg0[0, 0, %h_slice_offset, %w_slice_offset] [1, 16, 32, 16] [1, 1, 1, 1]
                    : tensor<1x16x128x128xf16, {order = #NHWC}> to tensor<1x16x32x16xf16, {order = #NHWC}>

                %input1_slice = tensor.extract_slice %arg1[0, 0, %h_slice_offset, %w_slice_offset] [1, 16, 32, 16] [1, 1, 1, 1]
                    : tensor<1x16x128x128xf16, {order = #NHWC}> to tensor<1x16x32x16xf16, {order = #NHWC}>

                %eltwise = VPU.NCE.Eltwise(%input0_slice, %input1_slice) {
                    op_type = #VPU.eltwise_type<ADD>,
                    ppe = #VPU.PPEInt<mode = <ADD>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
                } : tensor<1x16x32x16xf16, {order = #NHWC}>, tensor<1x16x32x16xf16, {order = #NHWC}> -> tensor<1x16x32x16xf16, {order = #NHWC}>

                %biased = VPU.NCE.Eltwise(%eltwise, %cst_bias) {
                    op_type = #VPU.eltwise_type<ADD>,
                    ppe = #VPU.PPEInt<mode = <ADD>, clamp_low = -2147483648 : i64, clamp_high = 2147483647 : i64,
                                      lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, quant_scale = [1.000000e+00], fp_prelu_alpha = 1.000000e+00 : f64>
                } : tensor<1x16x32x16xf16, {order = #NHWC}>, tensor<1x16x32x16xf16, {order = #NHWC}> -> tensor<1x16x32x16xf16, {order = #NHWC}>

                %updated = tensor.insert_slice %biased into %iter_buf_w[0, 0, %iv_h, %iv_w] [1, 16, 32, 16] [1, 1, 1, 1]
                    : tensor<1x16x32x16xf16, {order = #NHWC}> into tensor<1x16x128x128xf16, {order = #NHWC}>

                scf.yield %updated : tensor<1x16x128x128xf16, {order = #NHWC}>
            }

            scf.yield %result_w : tensor<1x16x128x128xf16, {order = #NHWC}>
        }

        return %result_h : tensor<1x16x128x128xf16, {order = #NHWC}>
    }
}

// CHECK-LABEL: @LoopInvariantCodeMotionTest

// CHECK: func.func @main_func0_static
// CHECK: %[[CST_BIAS:.*]] = const.Declare tensor<1x16x32x16xf16, {order = #NHWC}> = dense<5.000000e-01>
// CHECK: VPU.NCE.Eltwise({{%.+}}, {{%.+}}) {op_type = #VPU.eltwise_type<ADD>
// CHECK: VPU.NCE.Eltwise({{%.+}}, %[[CST_BIAS]]) {op_type = #VPU.eltwise_type<ADD>

// CHECK: func.func @main
// CHECK: scf.for
// CHECK: %[[H_OFFSET:.*]] = affine.max
// CHECK: scf.for
// CHECK: %[[W_OFFSET:.*]] = affine.max
// CHECK: tensor.extract_slice {{%.+}}[0, %[[H_OFFSET]], %[[W_OFFSET]], 0]
// CHECK: tensor.extract_slice {{%.+}}[0, %[[H_OFFSET]], %[[W_OFFSET]], 0]
// CHECK: Core.NestedCall @Module0::@main_func0_static
// CHECK: tensor.insert_slice
// CHECK: scf.yield
// CHECK: scf.yield
