//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --verify-diagnostics %s | FileCheck %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

func.func @ParsePrintDenseConst() -> tensor<2xf16> {
    %cst = const.Declare tensor<2xf16> = dense<[1.0, 2.0]> : tensor<2xf16>

    return %cst : tensor<2xf16>

    // CHECK-LABEL: @ParsePrintDenseConst
    // CHECK:       [[CST:%.+]] = const.Declare tensor<2xf16> = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf16>
    // CHECK:       return [[CST]]
}

// -----

func.func @ParsePrintDenseConstWithTransformation() -> tensor<1xf16> {
    %cst = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf32>, [#const.CastElemType<f16>]

    return %cst : tensor<1xf16>

    // CHECK-LABEL: @ParsePrintDenseConstWithTransformation
    // CHECK:       [[CST:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf32>, [#const.CastElemType<f16>]
    // CHECK:       return [[CST]]
}

// -----

func.func @ParsePrintDenseConstWithTransformations() -> tensor<3xf16> {
    %cst = const.Declare tensor<3xf16> = dense<1.0> : tensor<1xf32>, [#const.CastElemType<f16>, #const.Broadcast<0 : i64, 3 : i64>]

    return %cst : tensor<3xf16>

    // CHECK-LABEL: @ParsePrintDenseConstWithTransformations
    // CHECK:       [[CST:%.+]] = const.Declare tensor<3xf16> = dense<1.000000e+00> : tensor<1xf32>, [#const.CastElemType<f16>, #const.Broadcast<0 : i64, 3 : i64>]
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

    // CHECK-LABEL: @ParsePrintDenseResource()
    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x3x1x1xf32>
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

    // CHECK-LABEL: @ParsePrintDenseResourceSplat
    // CHECK:       [[CST:%.+]] = const.Declare tensor<2x3x1x1xf32>
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

    // CHECK-LABEL: @ParsePrintDenseConstRelocateWeightsTable
    // CHECK{LITERAL}:  #const.RelocateWeightsTable<weightsPtr=[0], sparsityPtr=56448 : i64, offsets=[0], weightsTableSize=0 : i64>
}

// -----

func.func @ParsePrintDenseConstRelocateWeightsTableLong() -> memref<16x1x1x4xsi32> {
    %cst = const.Declare memref<16x1x1x4xsi32> = dense<0> : tensor<16x1x1x4xsi32>, [#const.RelocateWeightsTable<weightsPtr=[65536], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=0 : i64, weightsElemBitSize=16 : i64>]

    return %cst : memref<16x1x1x4xsi32>

    // CHECK-LABEL: @ParsePrintDenseConstRelocateWeightsTableLong
    // CHECK{LITERAL}:  #const.RelocateWeightsTable<weightsPtr=[65536], sparsityPtr=16777215 : i64, offsets=[0], weightsTableSize=0 : i64, weightsElemBitSize=16 : i64>
}

// -----

func.func @ParsePrintSubByte() -> tensor<1x1x3x3xui4> {
  %cst = const.Declare tensor<1x1x3x3xui4> = dense_resource<subbyte> : tensor<1x1x3x3xsi4>, [#const.ConvertElemType<si8>, #const.CastElemType<si4>]
  return %cst : tensor<1x1x3x3xui4>

  // CHECK-LABEL: @ParsePrintSubByte
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

// -----

// Verify that a 1-byte i4 blob with different nibbles (0x12 → nibbles 0x2 and 0x1)
// is correctly rejected as non-splat, causing a size mismatch for a 6-element tensor
func.func @SubByteNonSplatDifferentNibbles() -> tensor<6xsi4> {
    // expected-error@+1 {{Size of dense resource buffer '1' in 'baseContent' doesn't match its type 'tensor<6xsi4>'}}
    %cst = const.Declare tensor<6xsi4> = dense_resource<nonsplat_i4> : tensor<6xsi4>
    return %cst : tensor<6xsi4>
}

{-#
  dialect_resources: {
    builtin: {
      nonsplat_i4: "0x0400000012"
    }
  }
#-}

// -----

// Verify that a 1-byte i4 blob with identical nibbles (0x33 → both nibbles 0x3)
// is correctly detected as splat, making 1-byte blob valid for a 6-element tensor
func.func @SubByteSplatSameNibbles() -> tensor<6xsi4> {
    %cst = const.Declare tensor<6xsi4> = dense_resource<splat_i4> : tensor<6xsi4>
    return %cst : tensor<6xsi4>

    // CHECK-LABEL: @SubByteSplatSameNibbles
    // CHECK:       [[CST:%.+]] = const.Declare tensor<6xsi4>
    // CHECK-SAME:       dense_resource<splat_i4>
    // CHECK:       return [[CST]]
}

{-#
  dialect_resources: {
    builtin: {
      splat_i4: "0x0400000033"
    }
  }
#-}

// -----

// Verify that a sub-byte tensor whose last byte contains padding bits that differ
// from the splat pattern is still accepted as valid. tensor<3xsi4> uses 2 bytes:
// byte[0]=0x33 (elements 0x3, 0x3), byte[1]=0xF3 (element 0x3, padding 0xF).
// Padding bits in the last byte must be ignored during splat detection.
func.func @SubByteSplatWithPaddingBits() -> tensor<3xsi4> {
    %cst = const.Declare tensor<3xsi4> = dense_resource<splat_padded_i4> : tensor<3xsi4>
    return %cst : tensor<3xsi4>

    // CHECK-LABEL: @SubByteSplatWithPaddingBits
    // CHECK:       [[CST:%.+]] = const.Declare tensor<3xsi4>
    // CHECK-SAME:       dense_resource<splat_padded_i4>
    // CHECK:       return [[CST]]
}

{-#
  dialect_resources: {
    builtin: {
      splat_padded_i4: "0x0400000033F3"
    }
  }
#-}

// -----

// Verify that a 1-byte i4 splat blob is accepted for a tensor with an odd number
// of elements that would normally require padding in the last byte. tensor<3xsi4>
// needs 2 bytes when uncompressed, but a splat representation uses only 1 byte.
// The splat detection must not attempt to drop_back(1) on a single-byte buffer.
func.func @SubByteSplatSingleByteOddElements() -> tensor<3xsi4> {
    %cst = const.Declare tensor<3xsi4> = dense_resource<splat_odd_i4> : tensor<3xsi4>
    return %cst : tensor<3xsi4>

    // CHECK-LABEL: @SubByteSplatSingleByteOddElements
    // CHECK:       [[CST:%.+]] = const.Declare tensor<3xsi4>
    // CHECK-SAME:       dense_resource<splat_odd_i4>
    // CHECK:       return [[CST]]
}

{-#
  dialect_resources: {
    builtin: {
      splat_odd_i4: "0x0400000033"
    }
  }
#-}

// -----

// Verify that a 1-byte u8 splat blob is accepted for a multi-element tensor.
// A single-byte buffer matching the element size is a valid splat representation.
func.func @ByteTypeSplatSingleByte() -> tensor<4xui8> {
    %cst = const.Declare tensor<4xui8> = dense_resource<splat_u8> : tensor<4xui8>
    return %cst : tensor<4xui8>

    // CHECK-LABEL: @ByteTypeSplatSingleByte
    // CHECK:       [[CST:%.+]] = const.Declare tensor<4xui8>
    // CHECK-SAME:       dense_resource<splat_u8>
    // CHECK:       return [[CST]]
}

{-#
  dialect_resources: {
    builtin: {
      splat_u8: "0x040000002A"
    }
  }
#-}
