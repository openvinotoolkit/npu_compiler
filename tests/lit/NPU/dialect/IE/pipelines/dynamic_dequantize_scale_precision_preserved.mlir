//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --convert-dynamic-dequantize-to-fake-quantize --convert-precision-to-fp --split-fake-quant %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

// Regression test for the vpux.fq_precise_scales/vpux.fq_precise_zero_points mechanism:
// ConvertPrecisionToFP rounds the FakeQuantize output-limit constants to f16 in between
// ConvertDynamicDequantizeToFakeQuantize and SplitFakeQuant. Recomputing the scale from those
// f16-rounded limits instead of using the cached fp32 attribute measurably loses precision
// (0.0123456791 vs 0.0123468137, ~9e-5 relative error for this input). SplitFakeQuant must end
// up re-emitting the original precise scale, not one recomputed from the rounded limits.

// CHECK-LABEL: @DynamicDequantizeScalePrecisionPreserved
func.func @DynamicDequantizeScalePrecisionPreserved(%arg0: tensor<1x8xf16>) -> tensor<1x8xf16> {
    %w = const.Declare tensor<1x8xsi8> = dense<[[-127, -91, -50, -10, 10, 50, 91, 127]]> : tensor<1x8xsi8>
    %scale = const.Declare tensor<1x1xf32> = dense<0.0123456789> : tensor<1x1xf32>
    %dq = IE.DynamicDequantize(%w, %scale) {dstElemType = f16, vpux.weights_import_dyn_dequant} :
        tensor<1x8xsi8>, tensor<1x1xf32> -> tensor<1x8xf16>
    %ret = IE.Add(%dq, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8xf16>, tensor<1x8xf16> -> tensor<1x8xf16>
    return %ret : tensor<1x8xf16>

    // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1xf32> = dense<0.0123456791> : tensor<1x1xf32>
    // CHECK-NOT: dense<0.012346813{{.*}}> : tensor<1x1xf32>
    // CHECK:     [[DQ:%.+]] = IE.DynamicDequantize(%{{.+}}, [[SCALE]]) {dstElemType = f16, vpux.synthetic_dyn_dequant}
    // CHECK:     return
}

//-----

// CHECK-LABEL: @DynamicDequantizeRoundedZeroPointPreserved
func.func @DynamicDequantizeRoundedZeroPointPreserved(%arg0: tensor<1x8xf16>) -> tensor<1x8xf16> {
    %w = const.Declare tensor<1x8xsi8> = dense<[[-127, -91, -50, -10, 10, 50, 91, 127]]> : tensor<1x8xsi8>
    %scale = const.Declare tensor<1x1xf32> = dense<0.0123456789> : tensor<1x1xf32>
    %zp = const.Declare tensor<1x1xsi8> = dense<0.99999994> : tensor<1x1xf32>, [#const.CastElemType<si8>]
    %dq = IE.DynamicDequantize(%w, %scale, %zp) {dstElemType = f16, vpux.weights_import_dyn_dequant} :
        tensor<1x8xsi8>, tensor<1x1xf32>, tensor<1x1xsi8> -> tensor<1x8xf16>
    %ret = IE.Add(%dq, %arg0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x8xf16>, tensor<1x8xf16> -> tensor<1x8xf16>
    return %ret : tensor<1x8xf16>

    // CHECK-DAG: [[SCALE:%.+]] = const.Declare tensor<1x1xf32> = dense<0.0123456791> : tensor<1x1xf32>
    // CHECK-DAG: [[ZP:%.+]] = const.Declare tensor<1x1xsi8> = dense<1> : tensor<1x1xsi8>
    // CHECK-NOT: const.Declare tensor<1x1xsi8> = dense<0> : tensor<1x1xsi8>
    // CHECK:     [[DQ:%.+]] = IE.DynamicDequantize(%{{.+}}, [[SCALE]], [[ZP]]) {dstElemType = f16, vpux.synthetic_dyn_dequant}
    // CHECK:     return
}
