//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --initial-low-precision-transformations --initial-transformations --adjust-precision --operation-conversion --adjust-for-vpu --scaleshift-processing --convert-const-dynamic-dequantize-to-dequantize %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Test is AI generated :)

// Batched MatMul with a raw si4 DynamicDequantize weight chain.
// Runs all registered IE pipeline stages up to (not including) --low-precision:
//   initial-low-precision-transformations → initial-transformations → adjust-precision →
//   operation-conversion → adjust-for-vpu → scaleshift-processing →
//   convert-const-dynamic-dequantize-to-dequantize
//
// Key transformations verified:
//   1. matmul-inputs-to-2d traces raw DynDeq chain and slices the batched MatMul
//   2. ConvertDQRawDataTypeToQuantized inserts QuantizeCast (si4 → quant)
//   3. ConvertMatMulToConv lowers to Convolution

// CHECK: !qElemType = !quant.uniform<i4:f16, 1.000000e+00>

// CHECK-LABEL: @BatchedMatMulRawDynDeqPipeline
module @BatchedMatMulRawDynDeqPipeline {
    net.NetworkInfo entryPoint : @main
    inputsInfo : {
        DataInfo "input" : tensor<1x2x1x256xf16>
        DataInfo "weights" : tensor<2x128x256xsi4>
    } outputsInfo : {
        DataInfo "output" : tensor<1x2x1x128xf16>
    }

    // CHECK: func.func @main([[INPUT:%.+]]: tensor<1x2x1x256xf16>, [[WEIGHTS:%.+]]: tensor<2x128x256xsi4>)
    func.func @main(%arg0: tensor<1x2x1x256xf16>, %arg1: tensor<2x128x256xsi4>) -> tensor<1x2x1x128xf16> {
        %cst_scale = const.Declare tensor<2x128x1xf16> = dense<1.0> : tensor<2x128x1xf16>

        %0 = IE.DynamicDequantize(%arg1, %cst_scale) {dstElemType = f16} : tensor<2x128x256xsi4>, tensor<2x128x1xf16> -> tensor<2x128x256xf16>
        %1 = IE.AffineReshape(%0) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 2, 128, 256]} : tensor<2x128x256xf16> -> tensor<1x2x128x256xf16>

        %2 = IE.MatMul(%arg0, %1) {transpose_b} : tensor<1x2x1x256xf16>, tensor<1x2x128x256xf16> -> tensor<1x2x1x128xf16>

        return %2 : tensor<1x2x1x128xf16>

        // Scales sliced per-batch from the original constant
        // CHECK-DAG:   [[SCALE_1:%.+]] = const.Declare tensor<1x128x1xf16> = dense<1.000000e+00> : tensor<2x128x1xf16>, [#const.SubView<[1, 0, 0], [1, 128, 1]>]
        // CHECK-DAG:   [[SCALE_0:%.+]] = const.Declare tensor<1x128x1xf16> = dense<1.000000e+00> : tensor<2x128x1xf16>, [#const.SubView<[0, 0, 0], [1, 128, 1]>]

        // Input sliced by batch dim
        // CHECK:       [[IN_0:%.+]] = IE.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 1, 256] : tensor<1x2x1x256xf16> to tensor<1x1x1x256xf16>
        // CHECK:       [[IN_1:%.+]] = IE.Slice [[INPUT]] [0, 1, 0, 0] [1, 1, 1, 256] : tensor<1x2x1x256xf16> to tensor<1x1x1x256xf16>

        // Weight batch 0: slice → QuantizeCast (raw si4 → quant) → DynamicDequantize
        // CHECK:       [[W_SLICE_0:%.+]] = IE.Slice [[WEIGHTS]] [0, 0, 0] [1, 128, 256] : tensor<2x128x256xsi4> to tensor<1x128x256xsi4>
        // CHECK:       [[W_QCAST_0:%.+]] = IE.QuantizeCast([[W_SLICE_0]]) {dstElemType = !qElemType} : tensor<1x128x256xsi4> -> tensor<1x128x256x!qElemType>
        // CHECK:       [[W_DQ_0:%.+]] = IE.DynamicDequantize([[W_QCAST_0]], [[SCALE_0]]) {dstElemType = f16} : tensor<1x128x256x!qElemType>, tensor<1x128x1xf16> -> tensor<1x128x256xf16>

        // Weight batch 1: same flow
        // CHECK:       [[W_SLICE_1:%.+]] = IE.Slice [[WEIGHTS]] [1, 0, 0] [1, 128, 256] : tensor<2x128x256xsi4> to tensor<1x128x256xsi4>
        // CHECK:       [[W_QCAST_1:%.+]] = IE.QuantizeCast([[W_SLICE_1]]) {dstElemType = !qElemType} : tensor<1x128x256xsi4> -> tensor<1x128x256x!qElemType>
        // CHECK:       [[W_DQ_1:%.+]] = IE.DynamicDequantize([[W_QCAST_1]], [[SCALE_1]]) {dstElemType = f16} : tensor<1x128x256x!qElemType>, tensor<1x128x1xf16> -> tensor<1x128x256xf16>

        // Input 0 reshaped for Conv and weights reshaped to 4D
        // CHECK:       [[IN_0_4D:%.+]] = IE.AffineReshape([[IN_0]])
        // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 256, 1, 1]} : tensor<1x1x1x256xf16> -> tensor<1x256x1x1xf16>
        // CHECK:       [[W_0_4D:%.+]] = IE.AffineReshape([[W_DQ_0]])
        // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [128, 256, 1, 1]} : tensor<1x128x256xf16> -> tensor<128x256x1x1xf16>
        // CHECK:       [[CONV_0:%.+]] = IE.Convolution([[IN_0_4D]], [[W_0_4D]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x1x1xf16>, tensor<128x256x1x1xf16> -> tensor<1x128x1x1xf16>

        // Input 1 reshaped for Conv and weights reshaped to 4D
        // CHECK:       [[IN_1_4D:%.+]] = IE.AffineReshape([[IN_1]])
        // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [0], [1, 2, 3]], shape_value = [1, 256, 1, 1]} : tensor<1x1x1x256xf16> -> tensor<1x256x1x1xf16>
        // CHECK:       [[W_1_4D:%.+]] = IE.AffineReshape([[W_DQ_1]])
        // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [128, 256, 1, 1]} : tensor<1x128x256xf16> -> tensor<128x256x1x1xf16>
        // CHECK:       [[CONV_1:%.+]] = IE.Convolution([[IN_1_4D]], [[W_1_4D]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x1x1xf16>, tensor<128x256x1x1xf16> -> tensor<1x128x1x1xf16>

        // Outputs reshaped and concatenated back to original batch shape
        // CHECK:       [[OUT_0:%.+]] = IE.AffineReshape([[CONV_0]])
        // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 128]} : tensor<1x128x1x1xf16> -> tensor<1x1x1x128xf16>
        // CHECK:       [[OUT_1:%.+]] = IE.AffineReshape([[CONV_1]])
        // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3], [3], [3]], shape_value = [1, 1, 1, 128]} : tensor<1x128x1x1xf16> -> tensor<1x1x1x128xf16>
        // CHECK:       [[CONCAT:%.+]] = IE.Concat([[OUT_0]], [[OUT_1]])
        // CHECK-SAME{LITERAL}:  {static_offsets = [[0, 0, 0, 0], [0, 1, 0, 0]]} : tensor<1x1x1x128xf16>, tensor<1x1x1x128xf16> -> tensor<1x2x1x128xf16>

        // CHECK:       return [[CONCAT]] : tensor<1x2x1x128xf16>
    }
}
