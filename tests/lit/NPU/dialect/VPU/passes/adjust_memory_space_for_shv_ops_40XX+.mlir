//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% compilation-mode=DefaultHW" --adjust-memory-space-for-shv-ops %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// Only conversions that cannot be done via DMA are lowered to SHAVE. This DMA feature is only supported starting with NPU4

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:  @SkipNonSHAVEConvert
// CHECK-SAME:      ([[INPUT_DDR:%.+]]: tensor<1x3x4x4xf32>)
func.func @SkipNonSHAVEConvert(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x4x4xf16> {
    %0 = VPU.Convert(%arg0) {dstElemType = f16} : tensor<1x3x4x4xf32> -> tensor<1x3x4x4xf16>
    return %0 : tensor<1x3x4x4xf16>

    // CHECK:       [[CONVERT:%.+]] = VPU.Convert([[INPUT_DDR]])
    // CHECK-SAME:    -> tensor<1x3x4x4xf16>
    // CHECK:       return [[CONVERT]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!BoundedType = tensor<1x1x1x4000xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>

// CHECK-LABEL: @AtanDmaAdjust
// CHECK-SAME:   [[INPUT_DDR:%.+]]: tensor<1x1x1x4000xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>)
func.func @AtanDmaAdjust(%arg0: !BoundedType) -> !BoundedType {
    %aux_buff = VPU.Empty : tensor<1x1x1x524288xui8>
    %out = VPU.AtanDma(%arg0, %aux_buff) : !BoundedType, tensor<1x1x1x524288xui8> -> !BoundedType
    return %out : !BoundedType

    // CHECK:     [[AUX_CMX:%.+]] = VPU.Empty : tensor<1x1x1x524288xui8, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    // CHECK:     [[OUT_DDR:%.+]] = VPU.AtanDma([[INPUT_DDR]], [[AUX_CMX]])
    // CHECK-SAME:    -> tensor<1x1x1x4000xf16, {dynamic_dims_mask = #const.OpaqueI64Elements<[0, 0, 0, 1]> : tensor<4xsi64>, order = #NCHW}>
    // CHECK:      return [[OUT_DDR]]
}
