//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --vpu-arch=%arch% --split-input-file --mlir-elide-elementsattrs-if-larger 8 --ws-monolithic-partial %s | FileCheck %s --strict-whitespace
// TODO: #-157476 Enable LIT test pipeline for other architectures
// REQUIRES: arch-NPU40XX

// CHECK-LABEL: @WeightsSeprationMode
{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x0000000400aa"
        }
    }
#-}

module @WeightsSeprationMode attributes {} {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x2x1x1xui8>
    } outputsInfo : {
        DataInfo "output" : tensor<1x2x1x1xui8>
    }

    func.func @main(%arg0: tensor<1x2x1x1xui8>) -> tensor<1x2x1x1xui8> {
        %cst = const.Declare tensor<1x2x1x1xui8> = dense_resource<vpux_ow_1> : tensor<1x2x1x1xui8>, [#const.Add<1.0 : f32>]
        return %cst : tensor<1x2x1x1xui8>
    }

    // Note: We mainly want to check that #const.Add is mapped to a VPU.Add and
    // don't care about any of the other functionality the pipelines perform.

    // CHECK:  func.func @wrapper_main([[ARG0:%.+]]: tensor<1x2x1x1xui8>) -> tensor<1x2x1x1xui8> {
    // CHECK:      [[CST:%.+]] = const.Declare tensor<1x2x1x1xui8> = dense_resource<vpux_ow_1> : tensor<1x2x1x1xui8>
    // CHECK:      [[ADD:%.+]] = VPU.Add
    // CHECK:      return
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x0000000400aa",
            vpux_ow_2: "0x0000000400aabbcc"
        }
    }
#-}

// CHECK-LABEL: @TwoConstants
module @TwoConstants attributes {} {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x2x1x1xui8>
    } outputsInfo : {
        DataInfo "output1" : tensor<1x2x1x1xui8>
        DataInfo "output2" : tensor<1x4x1x1xui8>
    }

    func.func @main(%arg0: tensor<1x2x1x1xui8>) -> (tensor<1x2x1x1xui8>, tensor<1x4x1x1xui8>) {
        %cst1 = const.Declare tensor<1x2x1x1xui8> = dense_resource<vpux_ow_1> : tensor<1x2x1x1xui8>,
            [#const.Add<1.0>]
        %cst2 = const.Declare tensor<1x4x1x1xui8> = dense_resource<vpux_ow_2> : tensor<1x4x1x1xui8>,
            [#const.Add<1.0>]
        return %cst1, %cst2 : tensor<1x2x1x1xui8>, tensor<1x4x1x1xui8>
    }

    // CHECK: func.func @wrapper_main([[ARG0:%.+]]: tensor<1x2x1x1xui8>) -> (tensor<1x2x1x1xui8>, tensor<1x4x1x1xui8>) {
    // CHECK:   [[CST1:%.+]] = const.Declare tensor<1x2x1x1xui8> = dense_resource<vpux_ow_1> : tensor<1x2x1x1xui8>
    // CHECK:   [[CST2:%.+]] = const.Declare tensor<1x4x1x1xui8> = dense_resource<vpux_ow_2> : tensor<1x4x1x1xui8>

    // CHECK:   [[CST1_ADD:%.+]] = VPU.Add
    // CHECK:   [[CST2_ADD:%.+]] = VPU.Add

    // Init output concatenated into single blob:

    // CHECK:   [[CST1_RCAST:%.+]] = Core.ReinterpretCast({{.+}}) : tensor<1x2x1x1xui8> -> tensor<2xi8>
    // CHECK:   [[CST2_RCAST:%.+]] = Core.ReinterpretCast({{.+}}) : tensor<1x4x1x1xui8> -> tensor<4xi8>
    // CHECK:   [[CONCAT:%.+]] = VPU.Concat([[CST1_RCAST]], [[CST2_RCAST]])

    // Main input sliced back into individual buffers

    // CHECK:   [[CST1_SLICE:%.+]] = VPU.Slice [[CONCAT]] [0] [2]
    // CHECK:   [[CST1_OUT:%.+]] = Core.ReinterpretCast([[CST1_SLICE]]) : tensor<2xi8> -> tensor<1x2x1x1xui8>
    // CHECK:   [[CST2_SLICE:%.+]] = VPU.Slice [[CONCAT]] [2] [4]
    // CHECK:   [[CST2_OUT:%.+]] = Core.ReinterpretCast([[CST2_SLICE]]) : tensor<4xi8> -> tensor<1x4x1x1xui8>

    // CHECK:   return [[CST1_OUT]], [[CST2_OUT]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x0000000400aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbcc00aabbdd"
        }
    }
#-}

module @DisableConvertQuantizeOpsToNceOpsForInitSchedule attributes {} {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output" : tensor<4x4x1x1xf16>
    }

    func.func @main() -> tensor<4x4x1x1xf32> {
        %weights = const.Declare tensor<4x4x1x1xf32> = dense_resource<vpux_ow_1> : tensor<4x4x1x1xf32>

        %input_low = const.Declare tensor<1x1x1x1xf32> = dense<-1.0> : tensor<1x1x1x1xf32>
        %input_high = const.Declare tensor<1x1x1x1xf32> = dense<1.0> : tensor<1x1x1x1xf32>
        %output_low = const.Declare tensor<1x1x1x1xf32> = dense<-2.0> : tensor<1x1x1x1xf32>
        %output_high = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
        %fq = IE.FakeQuantize(%weights, %input_low, %input_high, %output_low, %output_high) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
            levels = 256 : i64
        } : tensor<4x4x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<4x4x1x1xf32>

        return %fq : tensor<4x4x1x1xf32>
    }

// E#176434 Dequantize is not converted to NCE Eltwise due to accuracy issues.
// CHECK:       func.func @wrapper_main() -> tensor<4x4x1x1xf16>
// CHECK:           [[DEQUANTIZE:%.+]] = VPU.Dequantize({{.*}}) {dstElemType = f16}
// CHECK:           [[COPY:%.+]] = VPU.Copy([[DEQUANTIZE]]) : !VPU.DistributedTensor<4x4x1x1xf16, #NCHW, @CMX_NN
// CHECK-SAME:                          -> tensor<4x4x1x1xf16>
// CHECK:           return [[COPY]] : tensor<4x4x1x1xf16>
}
