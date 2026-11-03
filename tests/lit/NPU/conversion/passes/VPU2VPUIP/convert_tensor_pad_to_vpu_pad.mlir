//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Coverage for --convert-tensor-pad-to-vpu-pad: the tensor.pad subset (CONSTANT mode, static
// low/high padding, static or bounded-dynamic input) is converted to VPU.Pad while the IR is still
// tensor-based (before bufferization), reusing VPU.Pad type/bounds inference. Pads outside the
// supported subset are left untouched (see convert_tensor_pad_to_vpu_pad_unsupported).
//
// The pass is architecture agnostic: VPU.Pad has the same representation on every generation, so a
// single generic LIT covers all platforms. Which pipelines schedule the pass is a property of the
// compilation pipeline, not of the pass itself.

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --convert-tensor-pad-to-vpu-pad %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Supported: dynamic last dim carrying an upper-bound encoding (512), CONSTANT pad value 0.0, static
// asymmetric low/high. VPU.Pad::inferReturnTypes must pad the static dim in the shape (175+2+1 = 178)
// and the dynamic dim in the bounds (512+3+5 = 520).
// CHECK-LABEL: @PadDynamicBounded
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
func.func @PadDynamicBounded(
        %arg0: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>)
        -> tensor<1x32x178x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 178, 520]> : tensor<4xsi64>, order = #NHWC}> {
    %pad_val = arith.constant 0.0 : f16
    %padded = tensor.pad %arg0 low[0, 0, 2, 3] high[0, 0, 1, 5] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        tensor.yield %pad_val : f16
    } : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
      to tensor<1x32x178x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 178, 520]> : tensor<4xsi64>, order = #NHWC}>
    return %padded : tensor<1x32x178x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 178, 520]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK-NOT:  tensor.pad
    // CHECK:      [[PAD:%.+]] = VPU.Pad([[ARG0]])
    // CHECK-SAME:   mode = #IE.pad_mode<CONSTANT>
    // CHECK-SAME:   pad_value_attr = 0.000000e+00
    // CHECK-SAME:   pads_begin_attr = [0, 0, 2, 3]
    // CHECK-SAME:   pads_end_attr = [0, 0, 1, 5]
    // CHECK-SAME:   -> tensor<1x32x178x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 178, 520]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:      return [[PAD]]
}

// -----

// Static shape (no bounds): rewritten to VPU.Pad. Distinct low/high per dimension pin the
// getStaticLow/getStaticHigh -> pads_begin_attr/pads_end_attr mapping and the per-dimension result
// shape (1+2+4, 32+3+2, 175+5+1, 512+7+3).
// CHECK-LABEL: @PadStatic
func.func @PadStatic(%arg0: tensor<1x32x175x512xf16>) -> tensor<7x37x181x522xf16> {
    %pad_val = arith.constant 0.0 : f16
    %padded = tensor.pad %arg0 low[2, 3, 5, 7] high[4, 2, 1, 3] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        tensor.yield %pad_val : f16
    } : tensor<1x32x175x512xf16> to tensor<7x37x181x522xf16>
    return %padded : tensor<7x37x181x522xf16>

    // CHECK-NOT:  tensor.pad
    // CHECK:      [[PAD:%.+]] = VPU.Pad(%arg0)
    // CHECK-SAME:   mode = #IE.pad_mode<CONSTANT>
    // CHECK-SAME:   pads_begin_attr = [2, 3, 5, 7]
    // CHECK-SAME:   pads_end_attr = [4, 2, 1, 3]
    // CHECK-SAME:   -> tensor<7x37x181x522xf16>
    // CHECK:      return [[PAD]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Non-zero constant pad value (42.0) on a bounded dynamic input: the value must be carried onto
// VPU.Pad's pad_value_attr, and the result bounds must be input bounds + low + high = 512+1+1 = 514.
// CHECK-LABEL: @PadNonZeroValue
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
func.func @PadNonZeroValue(
        %arg0: tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>)
        -> tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}> {
    %pad_val = arith.constant 4.200000e+01 : f16
    %padded = tensor.pad %arg0 low[0, 0, 0, 1] high[0, 0, 0, 1] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        tensor.yield %pad_val : f16
    } : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
      to tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}>
    return %padded : tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK-NOT:  tensor.pad
    // CHECK:      [[PAD:%.+]] = VPU.Pad([[ARG0]])
    // CHECK-SAME:   mode = #IE.pad_mode<CONSTANT>
    // CHECK-SAME:   pad_value_attr = 4.200000e+01
    // CHECK-SAME:   pads_begin_attr = [0, 0, 0, 1]
    // CHECK-SAME:   pads_end_attr = [0, 0, 0, 1]
    // CHECK-SAME:   -> tensor<1x32x175x?xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:      return [[PAD]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Integer constant pad value on an integer-typed input: the value must be carried exactly onto
// VPU.Pad's f64 pad_value_attr (7 -> 7.0), not corrupted by a lossy/sign-extending conversion.
// CHECK-LABEL: @PadIntValue
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x175x?xi32, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
func.func @PadIntValue(
        %arg0: tensor<1x32x175x?xi32, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>)
        -> tensor<1x32x175x?xi32, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}> {
    %pad_val = arith.constant 7 : i32
    %padded = tensor.pad %arg0 low[0, 0, 0, 1] high[0, 0, 0, 1] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        tensor.yield %pad_val : i32
    } : tensor<1x32x175x?xi32, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
      to tensor<1x32x175x?xi32, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}>
    return %padded : tensor<1x32x175x?xi32, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK-NOT:  tensor.pad
    // CHECK:      [[PAD:%.+]] = VPU.Pad([[ARG0]])
    // CHECK-SAME:   pad_value_attr = 7.000000e+00
    // CHECK-SAME:   -> tensor<1x32x175x?xi32, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK:      return [[PAD]]
}

// -----

// 10^16 is above 2^53 yet exactly representable in f64, so it must convert: the criterion is the
// exactness of the f64 conversion, not the magnitude of the integer.
// CHECK-LABEL: @PadLargeExactIntValue
func.func @PadLargeExactIntValue(%arg0: tensor<1x4x4x4xi64>) -> tensor<1x4x4x6xi64> {
    %pad_val = arith.constant 10000000000000000 : i64
    %padded = tensor.pad %arg0 low[0, 0, 0, 1] high[0, 0, 0, 1] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        tensor.yield %pad_val : i64
    } : tensor<1x4x4x4xi64> to tensor<1x4x4x6xi64>
    return %padded : tensor<1x4x4x6xi64>

    // CHECK-NOT:  tensor.pad
    // CHECK:      [[PAD:%.+]] = VPU.Pad(%arg0)
    // CHECK-SAME:   pad_value_attr = 1.000000e+16
    // CHECK:      return [[PAD]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 5.000000e-01:128>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Quantized padding value: SCF tiling (createQuantizedPaddingValue) emits the storage zero-point as
// an integer arith.constant wrapped in a builtin.unrealized_conversion_cast to the quantized element
// type. The rewrite must unwrap the cast and carry the UNSIGNED storage value (u8 zero-point 128 ->
// 128.0, not the sign-extended -128.0).
// CHECK-LABEL: @PadQuantValue
func.func @PadQuantValue(
        %arg0: tensor<1x32x175x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>)
        -> tensor<1x32x175x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}> {
    %c = arith.constant 128 : i8
    %pad_val = builtin.unrealized_conversion_cast %c : i8 to !qElemType
    %padded = tensor.pad %arg0 low[0, 0, 0, 1] high[0, 0, 0, 1] {
      ^bb0(%a: index, %b: index, %c0: index, %d: index):
        tensor.yield %pad_val : !qElemType
    } : tensor<1x32x175x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
      to tensor<1x32x175x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}>
    return %padded : tensor<1x32x175x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK-NOT:  tensor.pad
    // CHECK:      [[PAD:%.+]] = VPU.Pad(%arg0)
    // CHECK-SAME:   pad_value_attr = 1.280000e+02
    // CHECK-SAME:   pads_begin_attr = [0, 0, 0, 1]
    // CHECK-SAME:   pads_end_attr = [0, 0, 0, 1]
    // CHECK:      return [[PAD]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 5.000000e-01:255>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Same as @PadQuantValue, but with a storage zero point above the signed range of the storage type:
// the i8 storage constant 255 must be read as UNSIGNED (255.0). A regression to signed extraction
// (getInt()/getSExtValue() on the signless storage attribute) would emit -1.0 and silently change the
// padding data, which this case pins down.
// CHECK-LABEL: @PadQuantValueAboveSignedRange
func.func @PadQuantValueAboveSignedRange(
        %arg0: tensor<1x32x175x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>)
        -> tensor<1x32x175x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}> {
    %c = arith.constant 255 : i8
    %pad_val = builtin.unrealized_conversion_cast %c : i8 to !qElemType
    %padded = tensor.pad %arg0 low[0, 0, 0, 1] high[0, 0, 0, 1] {
      ^bb0(%a: index, %b: index, %c0: index, %d: index):
        tensor.yield %pad_val : !qElemType
    } : tensor<1x32x175x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 512]> : tensor<4xsi64>, order = #NHWC}>
      to tensor<1x32x175x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}>
    return %padded : tensor<1x32x175x?x!qElemType, {bounds = #const.OpaqueI64Elements<[1, 32, 175, 514]> : tensor<4xsi64>, order = #NHWC}>

    // CHECK-NOT:  tensor.pad
    // CHECK:      [[PAD:%.+]] = VPU.Pad(%arg0)
    // CHECK-SAME:   pad_value_attr = 2.550000e+02
    // CHECK-SAME:   pads_begin_attr = [0, 0, 0, 1]
    // CHECK-SAME:   pads_end_attr = [0, 0, 0, 1]
    // CHECK:      return [[PAD]]
}

// -----

// nofold only forbids removing a pad whose source and result types are equal. With nonzero padding
// the rewritten VPU.Pad is not foldable, so the pad is converted.
// CHECK-LABEL: @PadNofoldNonZeroPadding
func.func @PadNofoldNonZeroPadding(%arg0: tensor<1x32x175x512xf16>) -> tensor<1x32x175x514xf16> {
    %pad_val = arith.constant 0.0 : f16
    %padded = tensor.pad %arg0 nofold low[0, 0, 0, 1] high[0, 0, 0, 1] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        tensor.yield %pad_val : f16
    } : tensor<1x32x175x512xf16> to tensor<1x32x175x514xf16>
    return %padded : tensor<1x32x175x514xf16>

    // CHECK-NOT:  tensor.pad
    // CHECK:      [[PAD:%.+]] = VPU.Pad(%arg0)
    // CHECK-SAME:   pads_begin_attr = [0, 0, 0, 1]
    // CHECK-SAME:   pads_end_attr = [0, 0, 0, 1]
    // CHECK:      return [[PAD]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// Dynamic low/high padding is never a VPU.Pad rewrite target: such a pad is left untouched, keeping
// its upstream tensor bufferization path.
// CHECK-LABEL: @PadDynamicPaddingUntouched
func.func @PadDynamicPaddingUntouched(%arg0: tensor<1x32x175x512xf16>, %low: index, %high: index)
        -> tensor<1x32x175x?xf16> {
    %pad_val = arith.constant 0.0 : f16
    %padded = tensor.pad %arg0 low[0, 0, 0, %low] high[0, 0, 0, %high] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        tensor.yield %pad_val : f16
    } : tensor<1x32x175x512xf16> to tensor<1x32x175x?xf16>
    return %padded : tensor<1x32x175x?xf16>

    // CHECK-NOT:  VPU.Pad
    // CHECK:      tensor.pad
}

// -----

// A static-padding tensor.pad inside a pure-host function must NOT be converted: VPU.Pad / a SW
// kernel must never be introduced into host code. The pass early-returns on config.pureHostCompileFunc
// functions, so the pad is left untouched.
// CHECK-LABEL: @PadInPureHostUntouched
func.func @PadInPureHostUntouched(%arg0: tensor<1x32x175x512xf16>) -> tensor<1x32x175x514xf16>
        attributes {config.pureHostCompileFunc} {
    %pad_val = arith.constant 0.0 : f16
    %padded = tensor.pad %arg0 low[0, 0, 0, 1] high[0, 0, 0, 1] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        tensor.yield %pad_val : f16
    } : tensor<1x32x175x512xf16> to tensor<1x32x175x514xf16>
    return %padded : tensor<1x32x175x514xf16>

    // CHECK-NOT:  VPU.Pad
    // CHECK:      tensor.pad
}

// -----

// The greedy driver folds a rank-0 pad away (tensor::PadOp::fold returns the source) before the
// rewrite pattern runs, so the pattern's empty-pad guard is never reached here.
// CHECK-LABEL: @PadRank0Folded
// CHECK-SAME:  [[ARG0:%.+]]: tensor<f16>
func.func @PadRank0Folded(%arg0: tensor<f16>) -> tensor<f16> {
    %pad_val = arith.constant 0.0 : f16
    %padded = tensor.pad %arg0 low[] high[] {
        tensor.yield %pad_val : f16
    } : tensor<f16> to tensor<f16>
    return %padded : tensor<f16>

    // CHECK-NOT:  tensor.pad
    // CHECK-NOT:  VPU.Pad
    // CHECK:      return [[ARG0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// MoveTensorOpsToCMX retypes the pad into CMX before the DefaultHW rewrite point, so VPU.Pad type
// inference must preserve mem_space.
// CHECK-LABEL: @PadInCMX
// CHECK-SAME:  [[ARG0:%.+]]: tensor<1x32x175x512xf16, {mem_space = @CMX_NN, order = #NHWC}>
func.func @PadInCMX(%arg0: tensor<1x32x175x512xf16, {mem_space = @CMX_NN, order = #NHWC}>)
        -> tensor<1x32x175x514xf16, {mem_space = @CMX_NN, order = #NHWC}> {
    %pad_val = arith.constant 0.0 : f16
    %padded = tensor.pad %arg0 low[0, 0, 0, 1] high[0, 0, 0, 1] {
      ^bb0(%a: index, %b: index, %c: index, %d: index):
        tensor.yield %pad_val : f16
    } : tensor<1x32x175x512xf16, {mem_space = @CMX_NN, order = #NHWC}>
      to tensor<1x32x175x514xf16, {mem_space = @CMX_NN, order = #NHWC}>
    return %padded : tensor<1x32x175x514xf16, {mem_space = @CMX_NN, order = #NHWC}>

    // CHECK-NOT:  tensor.pad
    // CHECK:      [[PAD:%.+]] = VPU.Pad([[ARG0]])
    // CHECK-SAME:   -> tensor<1x32x175x514xf16, {mem_space = @CMX_NN, order = #NHWC}>
    // CHECK:      return [[PAD]]
}
