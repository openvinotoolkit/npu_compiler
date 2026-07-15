//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --adjust-mem-permute-around-op %s | FileCheck %s
// REQUIRES: platform-NPU3720

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @AdjustForDynamicDequantize
// CHECK-SAME:  [[INPUT1:%.+]]: tensor<1x16x256x128xsi4>, [[INPUT2:%.+]]: tensor<16x256x1xf16>
func.func @AdjustForDynamicDequantize(%arg0: tensor<1x16x256x128xsi4>, %arg1: tensor<16x256x1xf16>) -> tensor<1x256x16x128xf16> {
    %0 = IE.QuantizeCast(%arg0) {dstElemType = !quant.uniform<i4:f16, 1.000000e+00>} : tensor<1x16x256x128xsi4> -> tensor<1x16x256x128x!quant.uniform<i4:f16, 1.000000e+00>>
    %1 = IE.AffineReshape(%arg1) {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 16, 256, 1]} : tensor<16x256x1xf16> -> tensor<1x16x256x1xf16>
    %2 = IE.DynamicDequantize(%0, %1) {dstElemType = f16} : tensor<1x16x256x128x!quant.uniform<i4:f16, 1.000000e+00>>, tensor<1x16x256x1xf16> -> tensor<1x16x256x128xf16>
    %3 = IE.MemPermute(%2) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<1x16x256x128xf16> -> tensor<1x256x16x128xf16>
    return %3 : tensor<1x256x16x128xf16>

    // CHECK:        [[QUANTIZE_CAST:%.+]] = IE.QuantizeCast([[INPUT1]])
    // CHECK:        [[AFFINE_RESHAPE:%.+]] = IE.AffineReshape([[INPUT2]])
    // CHECK:        [[DYNAMIC_DEQUANT:%.+]] = IE.DynamicDequantize([[QUANTIZE_CAST]], [[AFFINE_RESHAPE]])
    // CHECK:        [[MEM_PERMUTE:%.+]] = IE.MemPermute([[DYNAMIC_DEQUANT]])
    // CHECK:        return [[MEM_PERMUTE]] : tensor<1x256x16x128xf16>
}
