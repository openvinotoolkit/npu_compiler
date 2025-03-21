//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --verify-diagnostics %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX
func.func @ParsePrintDenseConst() -> tensor<2xf16> {
    %cst = const.Declare tensor<2xf16> = dense<[1.0, 2.0]> : tensor<2xf16>

    return %cst : tensor<2xf16>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<2xf16> = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf16>
    // CHECK:       return [[CST]]
}

// -----

func.func @ParsePrintDenseConstWithTransformation() -> tensor<1xf16> {
    %cst = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf32>, [#const.CastElemType<f16>]

    return %cst : tensor<1xf16>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf32>, [#const.CastElemType<f16>]
    // CHECK:       return [[CST]]
}

// -----

func.func @ParsePrintDenseConstWithTransformations() -> tensor<3xf16> {
    %cst = const.Declare tensor<3xf16> = dense<1.0> : tensor<1xf32>, [#const.CastElemType<f16>, #const.Broadcast<0 : i64, 3 : i64>]

    return %cst : tensor<3xf16>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<3xf16> = dense<1.000000e+00> : tensor<1xf32>, [#const.CastElemType<f16>, #const.Broadcast<0 : i64, 3 : i64>]
    // CHECK:       return [[CST]]
}

// -----

func.func @ParsePrintDenseConstWithInvalidConversion() -> tensor<2xi8> {
    // expected-error@+1 {{'Const.Declare' has mismatch in value element type 'f16' and result element type 'i8'}}
    %cst = const.Declare tensor<2xi8> = dense<1.0> : tensor<2xf32>, [#const.CastElemType<f16>, #const.Add<1.0>]

    return %cst : tensor<2xi8>
}

// -----

func.func @ParsePrintDenseConstWithInvalidBroadcast() -> tensor<3xf16> {
    // expected-error@+1 {{'Const.Declare' has mismatch in value shape '[2]' and result shape '[3]'}}
    %cst = const.Declare tensor<3xf16> = dense<1.0> : tensor<1xf16>, [#const.Broadcast<0 : i64, 2 : i64>]

    return %cst : tensor<3xf16>
}

// -----

func.func @ParsePrintDenseResource() -> tensor<1x3x1x1xf32> {
    %cst = const.Declare tensor<1x3x1x1xf32> = dense_resource<blob> : tensor<1x3x1x1xf32>

    return %cst : tensor<1x3x1x1xf32>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<1x3x1x1xf32>
    // CHECK-SAME:       dense_resource<blob>

    // CHECK:       return [[CST]]
}

{-#
  dialect_resources: {
    // Note: first 4 bytes in the dense_resource blob specify alignment
    builtin: {
      blob: "0x04000000010000000200000003000000"
    }
  }
#-}

// -----

func.func @ParsePrintDenseResourceNoAlignment() -> tensor<1x3x1x1xf32> {
    // expected-error@+1 {{Size of dense resource buffer '8' in 'baseContent' doesn't match its type 'tensor<1x3x1x1xf32>'}}
    %cst = const.Declare tensor<1x3x1x1xf32> = dense_resource<no_alignment_blob> : tensor<1x3x1x1xf32>

    return %cst : tensor<1x3x1x1xf32>
}

{-#
  dialect_resources: {
    builtin: {
      no_alignment_blob: "0x010000000200000003000000"
    }
  }
#-}

// -----

func.func @ParsePrintDenseResourceWrongDataSize() -> tensor<1x3x1x1xf16> {
    // expected-error@+1 {{Size of dense resource buffer '16' in 'baseContent' doesn't match its type 'tensor<1x3x1x1xf32>'}}
    %cst = const.Declare tensor<1x3x1x1xf16> = dense_resource<too_big_blob> : tensor<1x3x1x1xf32>, [#const.CastElemType<f16>]

    return %cst : tensor<1x3x1x1xf16>
}

{-#
  dialect_resources: {
    builtin: {
      too_big_blob: "0x0400000001000000020000000300000004000000"
    }
  }
#-}

// -----

func.func @ParsePrintDenseResourceSplat() -> tensor<2x3x1x1xf32> {
    %cst = const.Declare tensor<2x3x1x1xf32> = dense_resource<splat_blob> : tensor<2x3x1x1xf32>

    return %cst : tensor<2x3x1x1xf32>

    // CHECK:       [[CST:%.*]] = const.Declare tensor<2x3x1x1xf32>
    // CHECK-SAME:       dense_resource<splat_blob>

    // CHECK:       return [[CST]]
}

{-#
  dialect_resources: {
    builtin: {
      splat_blob: "0x0400000001000000"
    }
  }
#-}

// -----

func.func @ParsePrintDenseConstRelocateWeightsTable() -> memref<16x1x1x4xsi32> {
    %cst = const.Declare memref<16x1x1x4xsi32> = dense<0> : tensor<16x1x1x4xsi32>, [#const.RelocateWeightsTable<weightsPtr=[0], sparsityPtr=56448 : i64, offsets=[0], weightsTableSize=0 : i64>]

    return %cst : memref<16x1x1x4xsi32>

    // CHECK{LITERAL}:  #const.RelocateWeightsTable<weightsPtr=[0], sparsityPtr=56448 : i64, offsets=[0], weightsTableSize=0 : i64>
}

// -----

func.func @ParsePrintDenseConstRelocateWeightsTableLong() -> memref<16x1x1x4xsi32> {
    %cst = const.Declare memref<16x1x1x4xsi32> = dense<0> : tensor<16x1x1x4xsi32>, [#const.RelocateWeightsTable<weightsPtr=[65536], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=0 : i64, weightsElemBitSize=16 : i64>]

    return %cst : memref<16x1x1x4xsi32>

    // CHECK{LITERAL}:  #const.RelocateWeightsTable<weightsPtr=[65536], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=0 : i64, weightsElemBitSize=16 : i64>
}

// -----

func.func @ParsePrintSubByte() -> tensor<1x1x3x3xui4> {
  %cst = const.Declare tensor<1x1x3x3xui4> = dense_resource<subbyte> : tensor<1x1x3x3xsi4>, [#const.ConvertElemType<si8>, #const.CastElemType<si4>]
  return %cst : tensor<1x1x3x3xui4>

  // CHECK: [[CST:%.+]] = const.Declare tensor<1x1x3x3xui4>
  // CHECK: return [[CST]]
}

{-#
  dialect_resources: {
    // Note: first 4 bytes in the dense_resource blob specify alignment
    builtin: {
      subbyte: "0x040000001234567890"
    }
  }
#-}

// -----

func.func @ParsePrintInvalidSubByte() -> tensor<1x1x3x3xui4> {
  // expected-error@+1 {{Size of dense resource buffer '4' in 'baseContent' doesn't match its type 'tensor<1x1x3x3xsi4>'}}
  %cst = const.Declare tensor<1x1x3x3xui4> = dense_resource<subbyte> : tensor<1x1x3x3xsi4>, [#const.ConvertElemType<si8>, #const.CastElemType<si4>]
  return %cst : tensor<1x1x3x3xui4>
}

{-#
  dialect_resources: {
    // Note: first 4 bytes in the dense_resource blob specify alignment
    builtin: {
      subbyte: "0x0400000012345678"
    }
  }
#-}
