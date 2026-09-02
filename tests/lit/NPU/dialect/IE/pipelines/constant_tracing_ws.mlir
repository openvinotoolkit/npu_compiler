//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: env OV_NPU_LOG_LEVEL=LOG_TRACE vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW constant-tracing=true" --mlir-elide-elementsattrs-if-larger 8 --default-hw-mode-ie="enable-grouped-matmul=false" --construct-ws-analysis --introduce-init-function="ws-extraction-mode=gen-init" -o /dev/null %s 2>&1 | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_2bytes: "0x00000004aabb"
        }
    }
#-}

module @FuseConstDivideToMatMul {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x64x3x128xf16>
        DataInfo "input" : tensor<1x64x3x128xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<1x3x64x64xf16>
    }

    func.func @main(%arg0: tensor<1x3x64x128xf16>, %arg1: tensor<1x3x64x128xf16>) -> tensor<1x3x64x64xf16> {
        %cst_0 = const.Declare tensor<1xf16> = dense_resource<vpux_ow_2bytes> : tensor<1xf16>
        %cst_1 = const.Declare tensor<1xf16> = dense<2.550000e+02> : tensor<1xf16>
        %cst_16 = const.Declare tensor<1xf16> = dense<-8.01463317> : tensor<1xf16>
        %cst_17 = const.Declare tensor<1xf16> = dense<7.95201873> : tensor<1xf16>
        %cst_18 = const.Declare tensor<1xf16> = dense<2.460000e+02> : tensor<1xf16>
        %cst_fq = IE.FakeQuantize(%cst_18, %cst_0, %cst_1, %cst_16, %cst_17) {
            auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64
        } : tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16> -> tensor<1xf16>

        %28 = IE.MatMul(%arg0, %arg1) {transpose_b}
            : tensor<1x3x64x128xf16>, tensor<1x3x64x128xf16> -> tensor<1x3x64x64xf16>

        %29 = IE.Divide(%28, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<1x3x64x64xf16>, tensor<1xf16> -> tensor<1x3x64x64xf16>

        return %29 : tensor<1x3x64x64xf16>

        // CHECK: Trace: {{.*}}ConvertDivideToMultiply
        // CHECK: Transformation: #const.Reshape<[1, 1, 1, 1]>
        // CHECK: Trace: ConvertShapeTo4D -> apply-conversion
        // CHECK: Transformation: #const.Broadcast<1 : i64, 3 : i64>
        // CHECK: Trace: ConvertToScaleShift -> GreedyPatternRewriteIteration(1) -> `apply-pattern pattern: ConvertMultiplyToScaleShift
        // CHECK: Transformation: #const.Reshape<[3, 1, 1, 1]>
        // CHECK: Trace: ConvertScaleShiftToDW
        // CHECK: Transformation: #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>
        // CHECK: Trace: Canonicalizer -> GreedyPatternRewriteIteration(1)
        // CHECK: Transformation: #const.PadWithZero<[0, 0, 0, 0], [13, 0, 0, 0]>
        // CHECK: Trace: ExpandActivationChannels -> apply-conversion -> `apply-pattern pattern: GroupConvolutionRewriter

    }
}
