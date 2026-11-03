//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// --convert-tensor-pad-to-vpu-pad rewrites the supported tensor.pad subset and leaves everything
// else untouched, keeping the tensor bufferization path. One case per bail-out that is
// reachable from valid IR; empty low/high and a pad-value attribute that is neither float nor
// integer are not.
//
// The pass is architecture agnostic, hence no REQUIRES / platform suffix.

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --convert-tensor-pad-to-vpu-pad %s | FileCheck %s

// CHECK-LABEL: @PadUnboundedDynamicNotConverted
func.func @PadUnboundedDynamicNotConverted(%arg0: tensor<1x32x175x?xf16>) -> tensor<1x32x175x?xf16> {
    %pad_val = arith.constant 0.0 : f16
    %padded = tensor.pad %arg0 low[0, 0, 0, 1] high[0, 0, 0, 1] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        tensor.yield %pad_val : f16
    } : tensor<1x32x175x?xf16> to tensor<1x32x175x?xf16>
    return %padded : tensor<1x32x175x?xf16>

    // CHECK-NOT:  VPU.Pad
    // CHECK:      tensor.pad
}

// -----

// An index-dependent pad body (mirror/reflect/symmetric-style padding) has no constant padding
// value, so only IE::PadMode::CONSTANT is expressible.
// CHECK-LABEL: @PadIndexDependentBodyNotConverted
func.func @PadIndexDependentBodyNotConverted(%arg0: tensor<1x4x4x4xi32>) -> tensor<1x4x4x6xi32> {
    %padded = tensor.pad %arg0 low[0, 0, 0, 1] high[0, 0, 0, 1] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        %pad_val = arith.index_cast %d : index to i32
        tensor.yield %pad_val : i32
    } : tensor<1x4x4x4xi32> to tensor<1x4x4x6xi32>
    return %padded : tensor<1x4x4x6xi32>

    // CHECK-NOT:  VPU.Pad
    // CHECK:      tensor.pad
}

// -----

// A pad value defined outside the pad body is returned by getConstantPaddingValue() without being a
// constant; it cannot be folded into pad_value_attr.
// CHECK-LABEL: @PadNonConstantValueNotConverted
func.func @PadNonConstantValueNotConverted(%arg0: tensor<1x4x4x4xf16>, %pad_val: f16) -> tensor<1x4x4x6xf16> {
    %padded = tensor.pad %arg0 low[0, 0, 0, 1] high[0, 0, 0, 1] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        tensor.yield %pad_val : f16
    } : tensor<1x4x4x4xf16> to tensor<1x4x4x6xf16>
    return %padded : tensor<1x4x4x6xf16>

    // CHECK-NOT:  VPU.Pad
    // CHECK:      tensor.pad
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0019697112195632038:128>

// createQuantizedPaddingValue emits exactly the storage zero point (128 here). The cast carries no
// numeric semantics, so a different storage constant must not be reinterpreted as pad data.
// CHECK-LABEL: @PadQuantValueNotZeroPointNotConverted
func.func @PadQuantValueNotZeroPointNotConverted(%arg0: tensor<1x4x4x4x!qElemType>) -> tensor<1x4x4x6x!qElemType> {
    %c = arith.constant 7 : i8
    %pad_val = builtin.unrealized_conversion_cast %c : i8 to !qElemType
    %padded = tensor.pad %arg0 low[0, 0, 0, 1] high[0, 0, 0, 1] {
      ^bb0(%a: index, %b: index, %c0: index, %d: index):
        tensor.yield %pad_val : !qElemType
    } : tensor<1x4x4x4x!qElemType> to tensor<1x4x4x6x!qElemType>
    return %padded : tensor<1x4x4x6x!qElemType>

    // CHECK-NOT:  VPU.Pad
    // CHECK:      tensor.pad
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// A one-input builtin.unrealized_conversion_cast that is not the quantized-storage form is a
// semantics-free type bridge; its integer input must not be reused as the pad value.
// CHECK-LABEL: @PadNonQuantCastNotConverted
func.func @PadNonQuantCastNotConverted(
        %arg0: tensor<1x32x175x?xi16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>)
        -> tensor<1x32x175x?xi16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}> {
    %c = arith.constant 7 : i32
    %pad_val = builtin.unrealized_conversion_cast %c : i32 to i16
    %padded = tensor.pad %arg0 low[0, 0, 0, 1] high[0, 0, 0, 1] {
      ^bb0(%a: index, %b: index, %c0: index, %d: index):
        tensor.yield %pad_val : i16
    } : tensor<1x32x175x?xi16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
      to tensor<1x32x175x?xi16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}>
    return %padded : tensor<1x32x175x?xi16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK-NOT:  VPU.Pad
    // CHECK:      tensor.pad
}

// -----

// VPU.Pad's pad_value_attr is an f64: a wider pad value that does not round-trip exactly must be
// rejected. The value has to be spelled as a hexadecimal literal -- MLIR parses decimal float
// literals through `double`, so a decimal f80 constant is always exactly representable in f64.
// 0x3FFF8000000000000001 is 1.0 + 2^-63, whose trailing bit sits below the f64 ulp.
// CHECK-LABEL: @PadWideFloatInexactNotConverted
func.func @PadWideFloatInexactNotConverted(%arg0: tensor<1x4x4x4xf80>) -> tensor<1x4x4x6xf80> {
    %pad_val = arith.constant 0x3FFF8000000000000001 : f80
    %padded = tensor.pad %arg0 low[0, 0, 0, 1] high[0, 0, 0, 1] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        tensor.yield %pad_val : f80
    } : tensor<1x4x4x4xf80> to tensor<1x4x4x6xf80>
    return %padded : tensor<1x4x4x6xf80>

    // CHECK-NOT:  VPU.Pad
    // CHECK:      tensor.pad
}

// -----

// 2^53 + 1 is the smallest positive integer f64 cannot represent exactly; on pad_value_attr it
// would round to 2^53. Magnitude alone is not the criterion -- see @PadLargeExactIntValue.
// CHECK-LABEL: @PadIntAboveExactF64RangeNotConverted
func.func @PadIntAboveExactF64RangeNotConverted(%arg0: tensor<1x4x4x4xi64>) -> tensor<1x4x4x6xi64> {
    %pad_val = arith.constant 9007199254740993 : i64
    %padded = tensor.pad %arg0 low[0, 0, 0, 1] high[0, 0, 0, 1] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        tensor.yield %pad_val : i64
    } : tensor<1x4x4x4xi64> to tensor<1x4x4x6xi64>
    return %padded : tensor<1x4x4x6xi64>

    // CHECK-NOT:  VPU.Pad
    // CHECK:      tensor.pad
}

// -----

// 2^64 + 1 needs 65 mantissa bits, so it is inexact in f64. Values this wide must be rejected on
// the conversion status, not by narrowing the APInt first.
// CHECK-LABEL: @PadIntWiderThan64BitsInexactNotConverted
func.func @PadIntWiderThan64BitsInexactNotConverted(%arg0: tensor<1x4x4x4xi128>) -> tensor<1x4x4x6xi128> {
    %pad_val = arith.constant 18446744073709551617 : i128
    %padded = tensor.pad %arg0 low[0, 0, 0, 1] high[0, 0, 0, 1] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        tensor.yield %pad_val : i128
    } : tensor<1x4x4x4xi128> to tensor<1x4x4x6xi128>
    return %padded : tensor<1x4x4x6xi128>

    // CHECK-NOT:  VPU.Pad
    // CHECK:      tensor.pad
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// tensor.pad's verifier only compares ranks and static dims, so a result whose bounds do not match
// input bounds + low + high (600 instead of 512 + 1 + 1) is still valid IR. VPU.Pad infers 514, so
// the rewrite must bail out.
// CHECK-LABEL: @PadInferredBoundsMismatchNotConverted
func.func @PadInferredBoundsMismatchNotConverted(
        %arg0: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>)
        -> tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 600]> : tensor<4xsi64>, order = #NHWC}> {
    %pad_val = arith.constant 0.0 : f16
    %padded = tensor.pad %arg0 low[0, 0, 0, 1] high[0, 0, 0, 1] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        tensor.yield %pad_val : f16
    } : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
      to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 600]> : tensor<4xsi64>, order = #NHWC}>
    return %padded : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 600]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK-NOT:  VPU.Pad
    // CHECK:      tensor.pad
}

// -----

// nofold forbids removing a pad whose source and result types are equal. VPU::PadOp::fold removes
// exactly such a pad, so this one must stay a tensor.pad. See @PadNofoldNonZeroPadding for the
// nonzero-padding case, which carries no folding guarantee and is converted.
// CHECK-LABEL: @PadNofoldZeroPaddingNotConverted
func.func @PadNofoldZeroPaddingNotConverted(%arg0: tensor<1x32x175x512xf16>) -> tensor<1x32x175x512xf16> {
    %pad_val = arith.constant 0.0 : f16
    %padded = tensor.pad %arg0 nofold low[0, 0, 0, 0] high[0, 0, 0, 0] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        tensor.yield %pad_val : f16
    } : tensor<1x32x175x512xf16> to tensor<1x32x175x512xf16>
    return %padded : tensor<1x32x175x512xf16>

    // CHECK-NOT:  VPU.Pad
    // CHECK:      tensor.pad
}
