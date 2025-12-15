//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --constant-folding --mlir-print-elementsattrs-with-hex-if-larger=-1 %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>
#YXOI = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>

func.func @ConstFold() -> memref<16x3x1x1xf16, #YXOI> {
    %0 = const.Declare memref<16x3x1x1xf16, #YXOI> =
        dense<-1.0> : tensor<16x3x1x1xf32>,
        [
            #const.CastElemType<f16>,
            #const.CastElemType<ui8>,
            #const.CastElemType<!qElemType>,
            #const.Dequantize,
            #const.Reorder<#YXOI>
        ]

    return %0 : memref<16x3x1x1xf16, #YXOI>

    // CHECK:       [[CST:%.*]] = const.Declare memref<16x3x1x1xf16, #YXOI>
    // CHECK-SAME:       dense<
    // CHECK-SAME:       tensor<16x3x1x1xf16
    // CHECK-SAME:       {order = #YXOI}>
    // CHECK:       return [[CST]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.0039215686274509803>
#YXOI = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>

func.func @QuantConstFold() -> memref<16x3x1x1x!qElemType, #YXOI> {
    %0 = const.Declare memref<16x3x1x1x!qElemType, #YXOI> =
        dense<129> : tensor<16x3x1x1xui8>,
        [
            #const.CastElemType<!qElemType>,
            #const.Reorder<#YXOI>
        ]

    return %0 : memref<16x3x1x1x!qElemType, #YXOI>

    // CHECK:       [[CST:%.*]] = const.Declare memref<16x3x1x1x!qElemType, #YXOI>
    // CHECK-SAME:       dense<
    // CHECK-SAME:       tensor<16x3x1x1xui8
    // CHECK-SAME:       {order = #YXOI}>
    // CHECK:       return [[CST]]
}

// -----

func.func @I1SubviewConstFoldSplat() -> memref<1x16x3x3xi1> {
    %cst = const.Declare memref<1x16x3x3xi1> =
        dense<true> : tensor<1x32x3x3xi1>,
        [
            #const.SubView<[0, 16, 0, 0], [1, 16, 3, 3]>
        ]

    return %cst : memref<1x16x3x3xi1>

    // CHECK:   [[CST:%.*]] = const.Declare memref<1x16x3x3xi1> = dense<true> : tensor<1x16x3x3xi1>
    // CHECK:   return [[CST]]
}

// -----

// CHECK-LABEL: @I1SubviewConstFoldNonSplat1D
func.func @I1SubviewConstFoldNonSplat1D() -> memref<4xi1> {
    %cst = const.Declare memref<4xi1> =
        dense<[1, 0, 0, 1, 1, 1, 0, 1, 1, 0]> : tensor<10xi1>,
        [
            #const.SubView<[3], [4]>
        ]

    return %cst : memref<4xi1>

    // CHECK:           [[CST:%.*]] = const.Declare memref<4xi1>
    // CHECK-SAME:         = dense<[true, true, true, false]> : tensor<4xi1>
    // CHECK:           return [[CST]]
}


// -----

// CHECK-LABEL: @I1SubviewConstFoldNonSplat2D
func.func @I1SubviewConstFoldNonSplat2D() -> memref<1x2xi1> {
    %cst = const.Declare memref<1x2xi1> =
        dense<[[true, false, true, false]]> : tensor<1x4xi1>,
        [
            #const.SubView<[0, 2], [1, 2]>
        ]

    return %cst : memref<1x2xi1>

    // CHECK:           [[CST:%.*]] = const.Declare memref<1x2xi1>
    // CHECK{LITERAL}:      = dense<[[true, false]]> : tensor<1x2xi1>
    // CHECK:           return [[CST]]
}

// -----

// CHECK-LABEL: @I1SubviewConstFoldNonSplat3D
func.func @I1SubviewConstFoldNonSplat3D() -> memref<1x2x4xi1> {
    %cst = const.Declare memref<1x2x4xi1> =
        dense<[[[0, 0, 1, 1, 0, 1, 1, 1],
                [1, 0, 0, 1, 1, 1, 0, 1],
                [0, 0, 0, 1, 0, 1, 1, 1],
                [0, 1, 0, 1, 1, 0, 1, 0]]]> : tensor<1x4x8xi1>,
        [
            #const.SubView<[0, 2, 4], [1, 2, 4]>
        ]

    return %cst : memref<1x2x4xi1>

    // CHECK:           [[CST:%.*]] = const.Declare memref<1x2x4xi1>
    // CHECK{LITERAL}:      = dense<[[[false, true, true, true], [true, false, true, false]]]> : tensor<1x2x4xi1>
    // CHECK:           return [[CST]]
}

// -----

// CHECK-LABEL: @I1SubviewConstFoldNonSplat4D
func.func @I1SubviewConstFoldNonSplat4D() -> memref<1x16x1x1xi1> {
    %cst = const.Declare memref<1x16x1x1xi1> =
        dense<[[[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],
                [[1]],[[1]],[[1]],[[1]],[[1]],[[1]],[[1]],[[1]],
                [[0]],[[1]],[[0]],[[1]],[[0]],[[1]],[[0]],[[1]],
                [[1]],[[0]],[[1]],[[0]],[[1]],[[0]],[[1]],[[0]]]]> : tensor<1x32x1x1xi1>,
        [
            #const.SubView<[0, 16, 0, 0], [1, 16, 1, 1]>
        ]

    return %cst : memref<1x16x1x1xi1>

    // CHECK:               [[CST:%.*]] = const.Declare memref<1x16x1x1xi1> = dense<
    // CHECK-SAME{LITERAL}:     [[[[false]], [[true]], [[false]], [[true]], [[false]], [[true]], [[false]], [[true]],
    // CHECK-SAME{LITERAL}:       [[true]], [[false]], [[true]], [[false]], [[true]], [[false]], [[true]], [[false]]]]
    // CHECK-SAME:              >
    // CHECK-SAME:              tensor<1x16x1x1xi1>
    // CHECK:               return [[CST]]
}

// -----

// CHECK-LABEL: @I1SubviewConstFoldNonSplat5D
func.func @I1SubviewConstFoldNonSplat5D() -> memref<1x1x2x3x1xi1> {
    %cst = const.Declare memref<1x1x2x3x1xi1> =
        dense<[[[[[0], [1], [1], [0]], [[1], [1], [0], [0]], [[1], [0], [1], [0]]],
                [[[1], [1], [1], [0]], [[0], [0], [1], [1]], [[0], [1], [0], [1]]]]]> : tensor<1x2x3x4x1xi1>,
        [
            #const.SubView<[0, 1, 1, 1, 0], [1, 1, 2, 3, 1]>
        ]

    return %cst : memref<1x1x2x3x1xi1>

    // CHECK:               [[CST:%.*]] = const.Declare memref<1x1x2x3x1xi1>
    // CHECK-SAME{LITERAL}:     = dense<[[[[[false], [true], [true]], [[true], [false], [true]]]]]> : tensor<1x1x2x3x1xi1>
    // CHECK:               return [[CST]]
}

// -----

// CHECK-LABEL: @I4SubviewConstFoldSplat1D
func.func @I4SubviewConstFoldSplat1D() -> tensor<2xi8> {
    %cst = const.Declare tensor<2xi8> =
        dense_resource<blob> : tensor<4xi4>,
        [
            #const.SubView<[2], [2]>, #const.ConvertElemType<i8>
        ]

    return %cst : tensor<2xi8>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<2xi8> = dense<2> : tensor<2xi8>
    // CHECK:   return [[CST]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x040000002222"
    }
  }
#-}

// -----

// CHECK-LABEL: @I4SubviewConstFoldNonSplat1D
func.func @I4SubviewConstFoldNonSplat1D() -> tensor<2xi8> {
    %cst = const.Declare tensor<2xi8> =
        dense_resource<blob> : tensor<4xi4>,
        [
            #const.SubView<[2], [2]>, #const.ConvertElemType<i8>
        ]

    return %cst : tensor<2xi8>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<2xi8> = dense<[3, 4]> : tensor<2xi8>
    // CHECK:   return [[CST]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x040000002143"
    }
  }
#-}

// -----

// CHECK-LABEL: @I4SubviewConstFoldNonSplat2D
func.func @I4SubviewConstFoldNonSplat2D() -> tensor<1x2xi8> {
    %cst = const.Declare tensor<1x2xi8> =
        dense_resource<blob> : tensor<2x4xi4>,
        [
            #const.SubView<[1, 2], [1, 2]>, #const.ConvertElemType<i8>
        ]

    return %cst : tensor<1x2xi8>

    // CHECK:               [[CST:%.*]] = const.Declare tensor<1x2xi8>
    // CHECK-SAME{LITERAL}:     = dense<[[6, 7]]> : tensor<1x2xi8>
    // CHECK:               return [[CST]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x0400000010325476"
    }
  }
#-}

// -----

// CHECK-LABEL: @I4SubviewConstFoldNonSplat3D
func.func @I4SubviewConstFoldNonSplat3D() -> tensor<1x2x1xi8> {
    %cst = const.Declare tensor<1x2x1xi8> =
        dense_resource<blob> : tensor<2x4x1xi4>,
        [
            #const.SubView<[1, 2, 0], [1, 2, 1]>, #const.ConvertElemType<i8>
        ]

    return %cst : tensor<1x2x1xi8>

    // CHECK:               [[CST:%.*]] = const.Declare tensor<1x2x1xi8>
    // CHECK-SAME{LITERAL}:     = dense<[[[6], [7]]]> : tensor<1x2x1xi8>
    // CHECK:               return [[CST]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x0400000010325476"
    }
  }
#-}

// -----

// CHECK-LABEL: @I4SubviewConstFoldNonSplat4D
func.func @I4SubviewConstFoldNonSplat4D() -> tensor<1x2x1x1xi8> {
    %cst = const.Declare tensor<1x2x1x1xi8> =
        dense_resource<blob> : tensor<2x4x2x1xi4>,
        [
            #const.SubView<[1, 2, 1, 0], [1, 2, 1, 1]>, #const.ConvertElemType<i8>
        ]

    return %cst : tensor<1x2x1x1xi8>

    // CHECK:               [[CST:%.*]] = const.Declare tensor<1x2x1x1xi8>
    // CHECK-SAME{LITERAL}:     = dense<[[[[2]], [[0]]]]> : tensor<1x2x1x1xi8>
    // CHECK:               return [[CST]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x040000001032547667452301"
    }
  }
#-}

// -----

// CHECK-LABEL: @I4SubviewConstFoldNonSplat5D
func.func @I4SubviewConstFoldNonSplat5D() -> tensor<1x2x1x1x1xi8> {
    %cst = const.Declare tensor<1x2x1x1x1xi8> =
        dense_resource<blob> : tensor<2x4x2x1x1xi4>,
        [
            #const.SubView<[1, 2, 1, 0, 0], [1, 2, 1, 1, 1]>, #const.ConvertElemType<i8>
        ]

    return %cst : tensor<1x2x1x1x1xi8>

    // CHECK:               [[CST:%.*]] = const.Declare tensor<1x2x1x1x1xi8>
    // CHECK-SAME{LITERAL}:     = dense<[[[[[2]]], [[[0]]]]]> : tensor<1x2x1x1x1xi8>
    // CHECK:               return [[CST]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x040000001032547667452301"
    }
  }
#-}

// -----

// CHECK-LABEL: @broadcastNonSplatPreChannel
func.func @broadcastNonSplatPreChannel() -> tensor<1x9x1x1xi8> {
    %cst = const.Declare tensor<1x9x1x1xi8> = dense<[1, 2, 3]> : tensor<3xi8>, [#const.Reshape<[1, 3, 1, 1]>, #const.Broadcast<0 : i64, 3 : i64>, #const.Reshape<[1, 9, 1, 1]>]
    return %cst : tensor<1x9x1x1xi8>

    // CHECK:               [[CST:%.*]] = const.Declare tensor<1x9x1x1xi8> = dense<
    // CHECK-SAME{LITERAL}:     [[[[1]], [[2]], [[3]], [[1]], [[2]], [[3]], [[1]], [[2]], [[3]]]]
    // CHECK-SAME:              >
    // CHECK-SAME:              tensor<1x9x1x1xi8>
    // CHECK:               return [[CST]]
}

// -----

// CHECK-LABEL: @broadcastNonSplatPostChannel
func.func @broadcastNonSplatPostChannel() -> tensor<1x9x1x1xi8> {
    %cst = const.Declare tensor<1x9x1x1xi8> = dense<[1, 2, 3]> : tensor<3xi8>, [#const.Reshape<[1, 3, 1, 1]>, #const.Broadcast<2 : i64, 3 : i64>, #const.Reshape<[1, 9, 1, 1]>]
    return %cst : tensor<1x9x1x1xi8>

    // CHECK:               [[CST:%.*]] = const.Declare tensor<1x9x1x1xi8> = dense<
    // CHECK-SAME{LITERAL}:     [[[[1]], [[1]], [[1]], [[2]], [[2]], [[2]], [[3]], [[3]], [[3]]]]
    // CHECK-SAME:              >
    // CHECK-SAME:              tensor<1x9x1x1xi8>
    // CHECK:               return [[CST]]
}

// -----

// CHECK-LABEL: @broadcastNonSplatMiddleChannel
func.func @broadcastNonSplatMiddleChannel() -> tensor<1x3x2x2xi8> {
    %cst = const.Declare tensor<1x3x2x2xi8> = dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi8>, [#const.Reshape<[1, 3, 1, 2]>, #const.Broadcast<2 : i64, 2 : i64>]
    return %cst : tensor<1x3x2x2xi8>

    // CHECK:               [[CST:%.*]] = const.Declare tensor<1x3x2x2xi8> = dense<
    // CHECK-SAME{LITERAL}:     [[[[1, 2], [1, 2]],
    // CHECK-SAME{LITERAL}:       [[3, 4], [3, 4]],
    // CHECK-SAME{LITERAL}:       [[5, 6], [5, 6]]]]
    // CHECK-SAME:              >
    // CHECK-SAME:              tensor<1x3x2x2xi8>
    // CHECK:               return [[CST]]
}

// -----

// CHECK-LABEL: @broadcastNonSplatMiddleChannelNone1
func.func @broadcastNonSplatMiddleChannelNone1() -> tensor<1x3x4x2xi8> {
    %cst = const.Declare tensor<1x3x4x2xi8> = dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]> : tensor<12xi8>, [#const.Reshape<[1, 3, 2, 2]>, #const.Broadcast<2 : i64, 4 : i64>]
    return %cst : tensor<1x3x4x2xi8>
    // CHECK:               [[CST:%.*]] = const.Declare tensor<1x3x4x2xi8> = dense<
    // CHECK-SAME{LITERAL}:     [[[[1, 2], [3, 4], [1, 2], [3, 4]],
    // CHECK-SAME{LITERAL}:       [[5, 6], [7, 8], [5, 6], [7, 8]],
    // CHECK-SAME{LITERAL}:       [[9, 10], [11, 12], [9, 10], [11, 12]]]]
    // CHECK-SAME:              >
    // CHECK-SAME:              tensor<1x3x4x2xi8>
    // CHECK:               return [[CST]]
}

// -----

// CHECK-LABEL: @broadcastNonSplatTwoDims
func.func @broadcastNonSplatTwoDims() -> tensor<1x4x4x4xf16> {
    %cst = const.Declare tensor<1x4x4x4xf16> = dense<[[[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]]]>
            : tensor<1x1x1x4xf16>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 1, 4]>, #const.Broadcast<1 : i64, 4 : i64>, #const.Broadcast<2 : i64, 4 : i64>]
    return %cst : tensor<1x4x4x4xf16>
    // CHECK:               [[CST:%.*]] = const.Declare tensor<1x4x4x4xf16> = dense<
    // CHECK-SAME{LITERAL}:     [[[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00],
    // CHECK-SAME{LITERAL}:       [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00],
    // CHECK-SAME{LITERAL}:       [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00],
    // CHECK-SAME{LITERAL}:       [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00],
    // CHECK-SAME{LITERAL}:       [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]]]
    // CHECK-SAME:              >
    // CHECK-SAME:              tensor<1x4x4x4xf16>
    // CHECK:               return [[CST]]
}

// -----

// CHECK-LABEL: @broadcastNonSplatMultiDims
func.func @broadcastNonSplatMultiDims() -> tensor<1x4x4x4xf16> {
    %cst = const.Declare tensor<1x4x4x4xf16> = dense<[[[[1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00]]]]>
            : tensor<1x1x4x1xf16>, [#const.CastElemType<f16>, #const.Reshape<[1, 1, 4, 1]>, #const.Broadcast<1 : i64, 4 : i64>, #const.Broadcast<3 : i64, 4 : i64>]
    return %cst : tensor<1x4x4x4xf16>
    // CHECK:               [[CST:%.*]] = const.Declare tensor<1x4x4x4xf16> = dense<
    // CHECK-SAME{LITERAL}:     [[[[1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME{LITERAL}:       [3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00], [4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME{LITERAL}:       [3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00], [4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME{LITERAL}:       [3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00], [4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00]],
    // CHECK-SAME{LITERAL}:       [[1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00],
    // CHECK-SAME{LITERAL}:       [3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00], [4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00]]]]
    // CHECK-SAME:              >
    // CHECK-SAME:              tensor<1x4x4x4xf16>
    // CHECK:               return [[CST]]
}

// -----

!qElemType = !quant.uniform<i4:f16, 0.0039215686274509803>
#YXOI = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>

func.func @ConstFoldI4() -> memref<1x4x1x1x!qElemType, #YXOI> {
    %weights = const.Declare memref<1x4x1x1x!qElemType, #YXOI> = dense<[[[[1.000000e+00]], [[0.000000e+00]], [[1.000000e+00]], [[2.000000e+00]]]]> :
    tensor<1x4x1x1xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#YXOI>]
    return %weights : memref<1x4x1x1x!qElemType, #YXOI>
    // CHECK:       [[CST:%.*]] = const.Declare memref<1x4x1x1x!qElemType, #YXOI>
    // CHECK-SAME{LITERAL}:     dense<[[[[1, 33]]]]
    // CHECK-SAME:              tensor<1x1x1x2xui8
    // CHECK-SAME:              {order = #YXOI}>
    // CHECK:       return [[CST]]
}


// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!quantileFloatType = !QuantileFloat.quantileFloat<ui4:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}>
!qElemType = !quant.quantile<u4:f16:f16, {-1.000000e+00,-0.69619280099868774,-0.52507305145263672,-0.39491748809814453,-0.28444138169288635,-0.18477343022823334,-0.091050036251544952,0.000000e+00,0.07958029955625534,0.16093020141124725,0.24611230194568634,0.33791524171829224,0.44070982933044434,0.56261700391769409,0.72295683622360229,1.000000e+00}:0.07874348958333334>

// CHECK-LABEL: @ConstFoldNF4
func.func @ConstFoldNF4() -> memref<2x2x1x1x!qElemType, #NHWC> {
    %weights = const.Declare memref<2x2x1x1x!qElemType, #NHWC> = dense_resource<blob> : tensor<2x2x1x1xui4>, [
        #const.ConvertElemType<ui8>,
        #const.CastElemType<!quantileFloatType> ,
        #const.CastElemType<f16>,
        #const.CastElemType<ui4>,
        #const.CastElemType<!qElemType>,
        #const.Reorder<#NHWC>
        ]
    return %weights : memref<2x2x1x1x!qElemType, #NHWC>

    // CHECK:       [[CST:%.*]] = const.Declare memref<2x2x1x1x!qElemType, #NHWC>
    // CHECK-SAME{LITERAL}:     dense<[[[[16, 50]]]]> :
    // CHECK-SAME:              tensor<1x1x1x2xui8, {order = #NHWC}>, [#const.Reshape<[2, 2, 1, 1]>, #const.CastElemType<!qElemType>]
    // CHECK:       return [[CST]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x040000001032"
    }
  }
#-}

// -----

!qElemType = !quant.uniform<u8:f16:0, {1.0:128}>
func.func @ConstFoldDequantizeAxis0() -> memref<1x2x4x8xf16> {
    %0 = const.Declare memref<1x2x4x8xf16> =
        dense<1.0> : tensor<1x2x4x8xf16>,
        [
            #const.CastElemType<ui8>,
            #const.CastElemType<!qElemType>,
            #const.Dequantize
        ]

    return %0 : memref<1x2x4x8xf16>

    // CHECK:               [[CST:%.*]] = const.Declare memref<1x2x4x8xf16>
    // CHECK-SAME{LITERAL}: dense<-1.270000e+02>
    // CHECK-SAME:          tensor<1x2x4x8xf16>
    // CHECK:               return [[CST]]
}

// -----

!qElemType = !quant.uniform<u8:f16:1, {1.0:128, 0.5:64}>
func.func @ConstFoldDequantizeAxis1() -> memref<1x2x4x8xf16> {
    %0 = const.Declare memref<1x2x4x8xf16> =
        dense<1.0> : tensor<1x2x4x8xf16>,
        [
            #const.CastElemType<ui8>,
            #const.CastElemType<!qElemType>,
            #const.Dequantize
        ]

    return %0 : memref<1x2x4x8xf16>

    // CHECK:               [[CST:%.*]] = const.Declare memref<1x2x4x8xf16>
    // CHECK-SAME{LITERAL}: dense<[[[[-1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02], [-1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02], [-1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02], [-1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02]], [[-3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01], [-3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01], [-3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01], [-3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01]]]]>
    // CHECK-SAME:          tensor<1x2x4x8xf16>
    // CHECK:               return [[CST]]
}

// -----

!qElemType = !quant.uniform<u8:f16:2, {1.0:128, 0.5:64, 0.25:32, 0.125:16}>
func.func @ConstFoldDequantizeAxis2() -> memref<1x2x4x8xf16> {
    %0 = const.Declare memref<1x2x4x8xf16> =
        dense<1.0> : tensor<1x2x4x8xf16>,
        [
            #const.CastElemType<ui8>,
            #const.CastElemType<!qElemType>,
            #const.Dequantize
        ]

    return %0 : memref<1x2x4x8xf16>

    // CHECK:               [[CST:%.*]] = const.Declare memref<1x2x4x8xf16>
    // CHECK-SAME{LITERAL}: dense<[[[[-1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02], [-3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01], [-7.750000e+00, -7.750000e+00, -7.750000e+00, -7.750000e+00, -7.750000e+00, -7.750000e+00, -7.750000e+00, -7.750000e+00], [-1.875000e+00, -1.875000e+00, -1.875000e+00, -1.875000e+00, -1.875000e+00, -1.875000e+00, -1.875000e+00, -1.875000e+00]], [[-1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02, -1.270000e+02], [-3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01, -3.150000e+01], [-7.750000e+00, -7.750000e+00, -7.750000e+00, -7.750000e+00, -7.750000e+00, -7.750000e+00, -7.750000e+00, -7.750000e+00], [-1.875000e+00, -1.875000e+00, -1.875000e+00, -1.875000e+00, -1.875000e+00, -1.875000e+00, -1.875000e+00, -1.875000e+00]]]]>
    // CHECK-SAME:          tensor<1x2x4x8xf16>
    // CHECK:               return [[CST]]
}

// -----

!qElemType = !quant.uniform<u8:f16:3, {1.0:128, 0.5:64, 0.25:32, 0.125:16, 0.1:128, 0.2:128, 0.4:128, 0.8:128}>
func.func @ConstFoldDequantizeAxis3() -> memref<1x2x4x8xf16> {
    %0 = const.Declare memref<1x2x4x8xf16> =
        dense<1.0> : tensor<1x2x4x8xf16>,
        [
            #const.CastElemType<ui8>,
            #const.CastElemType<!qElemType>,
            #const.Dequantize
        ]

    return %0 : memref<1x2x4x8xf16>

    // CHECK:               [[CST:%.*]] = const.Declare memref<1x2x4x8xf16>
    // CHECK-SAME{LITERAL}: dense<[[[[-1.270000e+02, -3.150000e+01, -7.750000e+00, -1.875000e+00, -1.270310e+01, -2.540630e+01, -5.081250e+01, -1.016250e+02], [-1.270000e+02, -3.150000e+01, -7.750000e+00, -1.875000e+00, -1.270310e+01, -2.540630e+01, -5.081250e+01, -1.016250e+02], [-1.270000e+02, -3.150000e+01, -7.750000e+00, -1.875000e+00, -1.270310e+01, -2.540630e+01, -5.081250e+01, -1.016250e+02], [-1.270000e+02, -3.150000e+01, -7.750000e+00, -1.875000e+00, -1.270310e+01, -2.540630e+01, -5.081250e+01, -1.016250e+02]], [[-1.270000e+02, -3.150000e+01, -7.750000e+00, -1.875000e+00, -1.270310e+01, -2.540630e+01, -5.081250e+01, -1.016250e+02], [-1.270000e+02, -3.150000e+01, -7.750000e+00, -1.875000e+00, -1.270310e+01, -2.540630e+01, -5.081250e+01, -1.016250e+02], [-1.270000e+02, -3.150000e+01, -7.750000e+00, -1.875000e+00, -1.270310e+01, -2.540630e+01, -5.081250e+01, -1.016250e+02], [-1.270000e+02, -3.150000e+01, -7.750000e+00, -1.875000e+00, -1.270310e+01, -2.540630e+01, -5.081250e+01, -1.016250e+02]]]]>
    // CHECK-SAME:          tensor<1x2x4x8xf16>
    // CHECK:               return [[CST]]
}

// -----

!qElemType = !quant.uniform<u8:f16:4, {1.0:128, 0.5:64}>
func.func @ConstFoldDequantizeAxis4() -> memref<2x2x2x2x2xf16> {
    %0 = const.Declare memref<2x2x2x2x2xf16> =
        dense<1.0> : tensor<2x2x2x2x2xf16>,
        [
            #const.CastElemType<ui8>,
            #const.CastElemType<!qElemType>,
            #const.Dequantize
        ]

    return %0 : memref<2x2x2x2x2xf16>

    // CHECK:               [[CST:%.*]] = const.Declare memref<2x2x2x2x2xf16>
    // CHECK-SAME{LITERAL}: dense<[[[[[-1.270000e+02, -3.150000e+01], [-1.270000e+02, -3.150000e+01]], [[-1.270000e+02, -3.150000e+01], [-1.270000e+02, -3.150000e+01]]], [[[-1.270000e+02, -3.150000e+01], [-1.270000e+02, -3.150000e+01]], [[-1.270000e+02, -3.150000e+01], [-1.270000e+02, -3.150000e+01]]]], [[[[-1.270000e+02, -3.150000e+01], [-1.270000e+02, -3.150000e+01]], [[-1.270000e+02, -3.150000e+01], [-1.270000e+02, -3.150000e+01]]], [[[-1.270000e+02, -3.150000e+01], [-1.270000e+02, -3.150000e+01]], [[-1.270000e+02, -3.150000e+01], [-1.270000e+02, -3.150000e+01]]]]]>
    // CHECK-SAME:          tensor<2x2x2x2x2xf16>
    // CHECK:               return [[CST]]
}

// -----

func.func @RelocateFoldUnfusedConstantSingleCluster() -> (memref<5x1x1x4xsi32>, memref<2x1x1x4xsi32>) {
    // weightPtrStep = 1, sparsityPtrStep = 2
    %relocate = const.Declare memref<5x1x1x4xsi32> = dense<[[[[1, 2, 3, 3]]], [[[2, 4, 4, 4]]], [[[3, 6, 5, 5]]], [[[4, 8, 6, 6]]], [[[5, 10, 7, 7]]]]> : tensor<5x1x1x4xsi32>,
        [#const.RelocateWeightsTable<weightsPtr=[10], sparsityPtr=15 : i64, offsets=[0], weightsTableSize=80 : i64, weightsElemBitSize=16 : i64>]
    %subview_relocate = const.Declare memref<2x1x1x4xsi32> = dense<[[[[1, 2, 3, 3]]], [[[2, 4, 4, 4]]], [[[3, 6, 5, 5]]], [[[4, 8, 6, 6]]], [[[5, 10, 7, 7]]]]> : tensor<5x1x1x4xsi32>,
        [#const.SubView<[2, 0, 0, 0], [2, 1, 1, 4]>,
         #const.RelocateWeightsTable<weightsPtr=[10], sparsityPtr=15 : i64, offsets=[0], weightsTableSize=32 : i64, weightsElemBitSize=16 : i64, channelOffset=2 : i64, originalOC=0 : i64>]
    return %relocate, %subview_relocate : memref<5x1x1x4xsi32>, memref<2x1x1x4xsi32>

    // CHECK:               [[RELOCATE:%.+]] = const.Declare memref<5x1x1x4xsi32>
    // CHECK-SAME{LITERAL}:     dense<[[[[10, 15, 3, 3]]], [[[11, 17, 4, 4]]], [[[12, 19, 5, 5]]], [[[13, 21, 6, 6]]], [[[14, 23, 7, 7]]]]>
    // CHECK:               [[SUBVIEW_RELOCATE:%.+]] = const.Declare memref<2x1x1x4xsi32>
    // CHECK-SAME{LITERAL}:     dense<[[[[12, 19, 5, 5]]], [[[13, 21, 6, 6]]]]>
}

// -----

func.func @RelocateFoldFusedConstantSingleCluster() -> memref<1x1x1x16xsi32> {
    // weightPtrStep = 1, sparsityPtrStep = 2
    %0 = const.Declare memref<1x1x1x16xsi32> = dense<[[[[1, 2, 1, 2, 2, 4, 2, 3, 3, 6, 4, 5, 0, 0, 0, 0]]]]> : tensor<1x1x1x16xsi32>,
        [#const.RelocateWeightsTable<weightsPtr=[10], sparsityPtr=15 : i64, offsets=[0], weightsTableSize=48 : i64, weightsElemBitSize=16 : i64>]
    return %0 : memref<1x1x1x16xsi32>

    // CHECK: [[CST:%*]] = const.Declare memref<1x1x1x16xsi32>
    // CHECK-SAME{LITERAL}: dense<[[[[10, 15, 1, 2, 11, 17, 2, 3, 12, 19, 4, 5, 0, 0, 0, 0]]]]>
}

// -----

func.func @RelocateFoldUnfusedMultiCluster() -> (memref<4x1x1x4xsi32>, memref<2x1x1x4xsi32>) {
    %relocate = const.Declare memref<4x1x1x4xsi32> = dense<[[[[1, 2, 3, 3]]], [[[2, 4, 4, 4]]], [[[3, 6, 5, 5]]], [[[4, 8, 6, 6]]]]> : tensor<4x1x1x4xsi32>,
        [#const.RelocateWeightsTable<weightsPtr=[10, 20], sparsityPtr=5 : i64, offsets=[0, 2], weightsTableSize=64 : i64, weightsElemBitSize=16 : i64>]
    %subview_relocate = const.Declare memref<2x1x1x4xsi32> = dense<[[[[1, 2, 3, 3]]], [[[2, 4, 4, 4]]], [[[3, 6, 5, 5]]], [[[4, 8, 6, 6]]]]> : tensor<4x1x1x4xsi32>,
        [#const.SubView<[2, 0, 0, 0], [2, 1, 1, 4]>,
         #const.RelocateWeightsTable<weightsPtr=[20], sparsityPtr=5 : i64, offsets=[0], weightsTableSize=32 : i64, weightsElemBitSize=16 : i64, channelOffset=0 : i64, originalOC=0 : i64>]
    return %relocate, %subview_relocate : memref<4x1x1x4xsi32>, memref<2x1x1x4xsi32>

    // CHECK:               [[RELOCATE:%.+]] = const.Declare memref<4x1x1x4xsi32>
    // CHECK-SAME{LITERAL}:    dense<[[[[10, 5, 3, 3]]], [[[11, 7, 4, 4]]], [[[20, 5, 5, 5]]], [[[21, 7, 6, 6]]]]>
    // CHECK:               [[SUBVIEW_RELOCATE:%.+]] = const.Declare memref<2x1x1x4xsi32>
    // CHECK-SAME{LITERAL}:     dense<[[[[20, 5, 5, 5]]], [[[21, 7, 6, 6]]]]>
}

// -----

func.func @RelocateWithSparsityAndNon0ChannelOffsetFoldSingleCluster() -> memref<5x1x1x4xsi32> {
    // weightPtrStep = 20, sparsityPtrStep = 2
    %relocate = const.Declare memref<5x1x1x4xsi32> = dense<[[[[1, 2, 3, 3]]], [[[21, 4, 4, 4]]], [[[41, 6, 5, 5]]], [[[61, 8, 6, 6]]], [[[81, 10, 7, 7]]]]> : tensor<5x1x1x4xsi32>,
        [#const.RelocateWeightsTable<weightsPtr=[10], sparsityPtr=15 : i64, offsets=[0], weightsTableSize=80 : i64, weightsElemBitSize=16 : i64,
        weightsCompression=#VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<24> : tensor<10xi64>, alignment = 16 : i64>, channelOffset=5 : i64, originalOC=0 : i64>]
    return %relocate : memref<5x1x1x4xsi32>

    // CHECK:                   const.Declare memref<5x1x1x4xsi32>
    // CHECK-SAME{LITERAL}:     dense<[[[[250, 25, 3, 3]]], [[[298, 27, 4, 4]]], [[[346, 29, 5, 5]]], [[[394, 31, 6, 6]]], [[[442, 33, 7, 7]]]]>
}

// -----

func.func @DuplicationRelocateWithSparsityAndNon0ChannelOffsetFoldSingleCluster() -> memref<5x1x1x4xsi32> {
    // weightPtrStep = 20, sparsityPtrStep = 2
    %relocate = const.Declare memref<5x1x1x4xsi32> = dense<[[[[1, 2, 3, 3]]], [[[21, 4, 4, 4]]], [[[41, 6, 5, 5]]], [[[61, 8, 6, 6]]], [[[81, 10, 7, 7]]]]> : tensor<5x1x1x4xsi32>,
        [#const.RelocateWeightsTable<weightsPtr=[10], sparsityPtr=15 : i64, offsets=[0], weightsTableSize=80 : i64, weightsElemBitSize=16 : i64,
        weightsCompression=#VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<24> : tensor<10xi64>, alignment = 16 : i64>, originalOC=3 : i64>]
    return %relocate : memref<5x1x1x4xsi32>

    // CHECK:                   const.Declare memref<5x1x1x4xsi32>
    // CHECK-SAME{LITERAL}:     dense<[[[[10, 15, 3, 3]]], [[[58, 17, 4, 4]]], [[[106, 19, 5, 5]]], [[[10, 15, 6, 6]]], [[[10, 15, 7, 7]]]]>
}

// -----

func.func @DuplicationOffsetRelocateWithSparsityAndNon0ChannelOffsetFoldSingleCluster() -> memref<4x1x1x4xsi32> {
    // weightPtrStep = 20, sparsityPtrStep = 2
    %relocate = const.Declare memref<4x1x1x4xsi32> = dense<[[[[1, 2, 3, 3]]], [[[21, 4, 4, 4]]], [[[41, 6, 5, 5]]], [[[61, 8, 6, 6]]]]> : tensor<4x1x1x4xsi32>,
        [#const.RelocateWeightsTable<weightsPtr=[10], sparsityPtr=15 : i64, offsets=[0], weightsTableSize=64 : i64, weightsElemBitSize=16 : i64,
        weightsCompression=#VPUIP.SparsityCompressionAttr<axis = 0 : i64, numElems = dense<24> : tensor<10xi64>, alignment = 16 : i64>, channelOffset=1 : i64, originalOC=3 : i64>]
    return %relocate : memref<4x1x1x4xsi32>

    // CHECK:                   const.Declare memref<4x1x1x4xsi32>
    // CHECK-SAME{LITERAL}:     dense<[[[[58, 17, 3, 3]]], [[[106, 19, 4, 4]]], [[[10, 15, 5, 5]]], [[[10, 15, 6, 6]]]]>
}

// -----

func.func @RelocateFoldMultiClusterSliced1OrigChannel() -> memref<4x1x1x4xsi32> {
    %relocate = const.Declare memref<4x1x1x4xsi32> = dense<[[[[1, 2, 3, 3]]], [[[2, 4, 4, 4]]], [[[3, 6, 5, 5]]], [[[4, 8, 6, 6]]]]> : tensor<4x1x1x4xsi32>,
        [#const.RelocateWeightsTable<weightsPtr=[10], sparsityPtr=5 : i64, offsets=[0], weightsTableSize=64 : i64, weightsElemBitSize=16 : i64, originalOC = 1 : i64>]
    return %relocate : memref<4x1x1x4xsi32>

    // CHECK:               [[RELOCATE:%.+]] = const.Declare memref<4x1x1x4xsi32>
    // CHECK-SAME{LITERAL}:    dense<[[[[10, 5, 3, 3]]], [[[10, 5, 4, 4]]], [[[10, 5, 5, 5]]], [[[10, 5, 6, 6]]]]>
}

// -----

func.func @RelocateFoldSingleClusterSliced1OrigChannel() -> (memref<2x1x1x4xsi32>, memref<2x1x1x4xsi32>) {
    %relocate_lower = const.Declare memref<2x1x1x4xsi32> = dense<[[[[1, 2, 3, 3]]], [[[2, 4, 4, 4]]]]> : tensor<2x1x1x4xsi32>,
        [#const.RelocateWeightsTable<weightsPtr=[10], sparsityPtr=5 : i64, offsets=[0], weightsTableSize=32 : i64, weightsElemBitSize=16 : i64, channelOffset=0 : i64, originalOC=1 : i64>]
    %relocate_upper = const.Declare memref<2x1x1x4xsi32> = dense<[[[[3, 6, 5, 5]]], [[[4, 8, 6, 6]]]]> : tensor<2x1x1x4xsi32>,
        [#const.RelocateWeightsTable<weightsPtr=[10], sparsityPtr=5 : i64, offsets=[0], weightsTableSize=32 : i64, weightsElemBitSize=16 : i64, channelOffset=2 : i64, originalOC=1 : i64>]
    return %relocate_lower, %relocate_upper : memref<2x1x1x4xsi32>, memref<2x1x1x4xsi32>

    // CHECK:               [[RELOCATE_LOWER:%.+]] = const.Declare memref<2x1x1x4xsi32>
    // CHECK-SAME{LITERAL}:    dense<[[[[10, 5, 3, 3]]], [[[10, 5, 4, 4]]]]>
    // CHECK:               [[RELOCATE_UPPER:%.+]] = const.Declare memref<2x1x1x4xsi32>
    // CHECK-SAME{LITERAL}:     dense<[[[[10, 5, 5, 5]]], [[[10, 5, 6, 6]]]]>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i4:f16, 1.1534313725490195>

func.func @FoldSplatFusedConstant() -> memref<1x1x1x288xui8> {
    %cst = const.Declare memref<1x1x1x288xui8> =  dense<1> : tensor<16x1x1x4xsi32>,
        [#const.FuseWeights<tensor<1x1x1x288xui8>,
            weightsTable=<dense<1> : tensor<16x1x1x4xsi32>>,
            weights=<dense<1.000000e+00> : tensor<16x1x1x4xf16>, [#const.CastElemType<si4>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]>>
        ]

    return %cst : memref<1x1x1x288xui8>

    // CHECK: [[FUSED_CONSTANT:%.+]] = const.Declare memref<1x1x1x288xui8> =
    // CHECK-SAME{LITERAL}: dense<[[[[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17]]]]> : tensor<1x1x1x288xui8>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
!qElemType = !quant.uniform<i8<-127:127>:f16, 0.0078740157480314959>

func.func @FoldFusedSignedQuantizedWeightsMixedPrecision() -> memref<1x1x1x512xui8> {
    %cst = const.Declare memref<1x1x1x512xui8> = dense<[[[[0, 0, 1006699012, 0]]], [[[16, 0, 1006699012, 0]]], [[[32, 0, 1006699012, 0]]], [[[48, 0, 1006699012, 0]]], [[[64, 0, 1006699012, 0]]], [[[80, 0, 1006699012, 0]]], [[[96, 0, 1006699012, 0]]], [[[112, 0, 1006699012, 0]]], [[[128, 0, 1006699012, 0]]], [[[144, 0, 1006699012, 0]]], [[[160, 0, 1006699012, 0]]], [[[176, 0, 1006699012, 0]]], [[[192, 0, 1006699012, 0]]], [[[208, 0, 1006699012, 0]]], [[[224, 0, 1006699012, 0]]], [[[240, 0, 1006699012, 0]]]]> : tensor<16x1x1x4xsi32>,
        [#const.FuseWeights<tensor<1x1x1x512xui8>,
            weightsTable=<dense<[[[[0, 0, 1006699012, 0]]], [[[16, 0, 1006699012, 0]]], [[[32, 0, 1006699012, 0]]], [[[48, 0, 1006699012, 0]]], [[[64, 0, 1006699012, 0]]], [[[80, 0, 1006699012, 0]]], [[[96, 0, 1006699012, 0]]], [[[112, 0, 1006699012, 0]]], [[[128, 0, 1006699012, 0]]], [[[144, 0, 1006699012, 0]]], [[[160, 0, 1006699012, 0]]], [[[176, 0, 1006699012, 0]]], [[[192, 0, 1006699012, 0]]], [[[208, 0, 1006699012, 0]]], [[[224, 0, 1006699012, 0]]], [[[240, 0, 1006699012, 0]]]]> : tensor<16x1x1x4xsi32>>,
            weights=<dense<[[[[-1.000000e+01]], [[-1.659180e+00]], [[9.945310e+00]], [[4.406250e+00]], [[8.648430e+00]], [[-1.000000e+01]], [[-7.437500e+00]], [[-3.953130e+00]], [[9.984370e+00]], [[-7.066400e+00]], [[-5.277340e+00]], [[-8.156250e+00]], [[-2.068360e+00]], [[-6.273440e+00]], [[-2.242190e+00]], [[-3.087890e+00]]], [[[3.394530e+00]], [[-2.064450e+00]], [[8.710930e+00]], [[7.763670e-01]], [[6.925780e+00]], [[-1.616210e+00]], [[-3.734380e+00]], [[3.705080e+00]], [[4.909670e-01]], [[-5.910150e+00]], [[-1.130860e+00]], [[7.562500e+00]], [[-5.410150e+00]], [[-9.453120e+00]], [[6.884770e-01]], [[3.410160e+00]]], [[[8.281250e+00]], [[-1.654300e+00]], [[-8.559570e-01]], [[1.173830e+00]], [[-1.385740e+00]], [[-7.191400e+00]], [[8.781250e+00]], [[-6.039060e+00]], [[5.566400e+00]], [[6.015630e+00]], [[4.320310e+00]], [[9.367180e+00]], [[6.054690e+00]], [[-3.732420e+00]], [[-8.140630e+00]], [[3.845700e+00]]], [[[3.630370e-01]], [[7.527340e+00]], [[7.300780e+00]], [[7.890630e+00]], [[6.582030e+00]], [[-8.296880e+00]], [[6.593750e+00]], [[-9.218750e+00]], [[-4.539060e+00]], [[-6.601560e+00]], [[-8.812500e+00]], [[7.562500e+00]], [[3.410160e+00]], [[-8.031250e+00]], [[1.861330e+00]], [[-1.578130e+00]]], [[[3.433590e+00]], [[9.156250e+00]], [[-1.764650e+00]], [[6.630860e-01]], [[-6.050780e+00]], [[3.837890e+00]], [[-4.207030e+00]], [[-3.689450e+00]], [[-7.156250e+00]], [[3.730470e+00]], [[5.667960e+00]], [[6.691400e+00]], [[-1.749020e+00]], [[-9.632810e+00]], [[-9.320310e+00]], [[5.003910e+00]]], [[[2.480470e+00]], [[9.773430e+00]], [[3.212890e+00]], [[4.964840e+00]], [[-4.031250e+00]], [[-4.390630e+00]], [[-1.077150e+00]], [[5.785150e+00]], [[-5.558590e+00]], [[-7.933590e+00]], [[-8.531250e+00]], [[-1.041990e+00]], [[-6.152340e-01]], [[8.171880e+00]], [[-8.078130e+00]], [[-4.128910e+00]]], [[[8.070310e+00]], [[-4.246090e+00]], [[-7.609380e+00]], [[-7.398430e+00]], [[4.960940e-01]], [[-9.609370e+00]], [[-8.328130e+00]], [[3.576170e+00]], [[8.335930e+00]], [[-5.765630e+00]], [[8.210930e+00]], [[-4.687500e+00]], [[-4.019530e+00]], [[-1.685790e-01]], [[1.687500e+00]], [[-8.929680e+00]]], [[[1.318360e+00]], [[1.482420e+00]], [[2.279300e+00]], [[-7.066400e+00]], [[9.132810e+00]], [[1.786130e+00]], [[-4.781250e+00]], [[3.996090e+00]], [[-5.378900e+00]], [[-7.953130e+00]], [[6.689450e-01]], [[-1.718750e+00]], [[9.000000e+00]], [[3.888670e+00]], [[-1.387940e-01]], [[-1.716800e+00]]], [[[8.120110e-01]], [[-9.000000e+00]], [[5.308590e+00]], [[7.177730e-01]], [[-9.093750e+00]], [[3.275390e+00]], [[-7.199210e+00]], [[2.978520e-01]], [[5.847650e+00]], [[8.890620e+00]], [[-9.406250e+00]], [[1.731450e+00]], [[7.664060e+00]], [[8.070310e+00]], [[8.159170e-01]], [[-7.250000e+00]]], [[[-1.040040e+00]], [[-7.214840e+00]], [[7.843750e+00]], [[6.148440e+00]], [[-2.449220e+00]], [[-2.046880e+00]], [[7.685550e-01]], [[-6.691400e+00]], [[3.046880e+00]], [[8.546870e+00]], [[-2.775390e+00]], [[-3.044920e+00]], [[1.419920e+00]], [[5.015630e+00]], [[2.755860e+00]], [[4.519530e+00]]], [[[-7.472650e+00]], [[7.667960e+00]], [[3.804690e+00]], [[2.472660e+00]], [[2.955080e+00]], [[5.019530e+00]], [[-2.921880e+00]], [[-3.021480e+00]], [[5.265630e+00]], [[-4.601560e+00]], [[-2.869140e+00]], [[7.917960e+00]], [[5.054690e+00]], [[-1.438480e+00]], [[7.625000e+00]], [[9.296870e+00]]], [[[-9.765620e+00]], [[3.269530e+00]], [[-3.781130e-02]], [[2.433590e+00]], [[-8.523430e+00]], [[-7.707030e+00]], [[5.738280e+00]], [[8.992180e+00]], [[-8.718750e+00]], [[-1.001950e+00]], [[-2.894530e+00]], [[1.567380e+00]], [[8.835930e+00]], [[-1.836910e+00]], [[-2.404300e+00]], [[-5.257810e+00]]], [[[5.257810e+00]], [[8.070310e+00]], [[5.433590e+00]], [[1.473630e+00]], [[-3.972660e+00]], [[-9.945310e+00]], [[5.453130e+00]], [[2.343750e+00]], [[-6.941400e+00]], [[-3.466800e+00]], [[1.572270e+00]], [[5.410150e-01]], [[-9.820310e+00]], [[7.718750e+00]], [[4.179690e+00]], [[-2.855470e+00]]], [[[-5.874020e-01]], [[8.171880e+00]], [[5.292970e+00]], [[2.466800e+00]], [[-6.523440e-01]], [[-9.679680e+00]], [[-4.621090e+00]], [[8.585930e+00]], [[6.632810e+00]], [[3.818360e+00]], [[1.026370e+00]], [[9.945310e+00]], [[-8.601560e+00]], [[-6.554690e+00]], [[-5.502930e-01]], [[-7.257810e+00]]], [[[4.855470e+00]], [[8.648430e+00]], [[-6.160150e+00]], [[3.935550e+00]], [[-7.138670e-01]], [[-8.679680e+00]], [[-5.394530e+00]], [[5.109380e+00]], [[1.649170e-01]], [[5.078130e+00]], [[-5.828130e+00]], [[8.460930e+00]], [[-9.015620e+00]], [[4.230470e+00]], [[3.771970e-01]], [[-7.515630e+00]]], [[[-6.554690e+00]], [[-9.601560e+00]], [[-2.074220e+00]], [[-9.476560e+00]], [[-7.851560e+00]], [[-9.437500e+00]], [[1.961670e-01]], [[-5.074220e+00]], [[-7.957030e+00]], [[7.199210e+00]], [[-4.226560e+00]], [[7.768550e-01]], [[-5.363280e+00]], [[1.056640e+00]], [[9.351560e+00]], [[1.000000e+01]]]]> : tensor<16x16x1x1xf16>, [#const.CastElemType<si8>, #const.CastElemType<!qElemType>, #const.Reorder<#NHWC>]>>
        ]
    return %cst : memref<1x1x1x512xui8>

    // CHECK: [[FUSED_CONSTANT:%.+]] = const.Declare memref<1x1x1x512xui8> =
    // CHECK-SAME{LITERAL}: dense<[[[[0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1, 60, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1, 60, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1, 60, 0, 0, 0, 0, 48, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1, 60, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1, 60, 0, 0, 0, 0, 80, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1, 60, 0, 0, 0, 0, 96, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1, 60, 0, 0, 0, 0, 112, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1, 60, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1, 60, 0, 0, 0, 0, 144, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1, 60, 0, 0, 0, 0, 160, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1, 60, 0, 0, 0, 0, 176, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1, 60, 0, 0, 0, 0, 192, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1, 60, 0, 0, 0, 0, 208, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1, 60, 0, 0, 0, 0, 224, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1, 60, 0, 0, 0, 0, 240, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1, 60, 0, 0, 0, 0, 246, 255, 9, 4, 8, 246, 249, 253, 9, 249, 251, 248, 254, 250, 254, 253, 3, 254, 8, 0, 6, 255, 253, 3, 0, 251, 255, 7, 251, 247, 0, 3, 8, 255, 0, 1, 255, 249, 8, 250, 5, 6, 4, 9, 6, 253, 248, 3, 0, 7, 7, 7, 6, 248, 6, 247, 252, 250, 248, 7, 3, 248, 1, 255, 3, 9, 255, 0, 250, 3, 252, 253, 249, 3, 5, 6, 255, 247, 247, 5, 2, 9, 3, 4, 252, 252, 255, 5, 251, 249, 248, 255, 0, 8, 248, 252, 8, 252, 249, 249, 0, 247, 248, 3, 8, 251, 8, 252, 252, 0, 1, 248, 1, 1, 2, 249, 9, 1, 252, 3, 251, 249, 0, 255, 9, 3, 0, 255, 0, 247, 5, 0, 247, 3, 249, 0, 5, 8, 247, 1, 7, 8, 0, 249, 255, 249, 7, 6, 254, 254, 0, 250, 3, 8, 254, 253, 1, 5, 2, 4, 249, 7, 3, 2, 2, 5, 254, 253, 5, 252, 254, 7, 5, 255, 7, 9, 247, 3, 0, 2, 248, 249, 5, 8, 248, 255, 254, 1, 8, 255, 254, 251, 5, 8, 5, 1, 253, 247, 5, 2, 250, 253, 1, 0, 247, 7, 4, 254, 0, 8, 5, 2, 0, 247, 252, 8, 6, 3, 1, 9, 248, 250, 0, 249, 4, 8, 250, 3, 0, 248, 251, 5, 0, 5, 251, 8, 247, 4, 0, 249, 250, 247, 254, 247, 249, 247, 0, 251, 249, 7, 252, 0, 251, 1, 9, 10]]]]> : tensor<1x1x1x512xui8>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func.func @FoldFusedWeightWithMajorityOfF16Type() -> memref<1x1x1x384xf16> {
    %cst = const.Declare memref<1x1x1x384xf16> = dense<[[[[0, 0, 1006699012, 0]]], [[[16, 0, 1006699012, 0]]], [[[32, 0, 1006699012, 0]]], [[[48, 0, 1006699012, 0]]], [[[64, 0, 1006699012, 0]]], [[[80, 0, 1006699012, 0]]], [[[96, 0, 1006699012, 0]]], [[[112, 0, 1006699012, 0]]], [[[128, 0, 1006699012, 0]]], [[[144, 0, 1006699012, 0]]], [[[160, 0, 1006699012, 0]]], [[[176, 0, 1006699012, 0]]], [[[192, 0, 1006699012, 0]]], [[[208, 0, 1006699012, 0]]], [[[224, 0, 1006699012, 0]]], [[[240, 0, 1006699012, 0]]]]> : tensor<16x1x1x4xsi32>,
        [#const.FuseWeights<tensor<1x1x1x384xf16>,
            weightsTable=<dense<[[[[0, 0, 1006699012, 0]]], [[[16, 0, 1006699012, 0]]], [[[32, 0, 1006699012, 0]]], [[[48, 0, 1006699012, 0]]], [[[64, 0, 1006699012, 0]]], [[[80, 0, 1006699012, 0]]], [[[96, 0, 1006699012, 0]]], [[[112, 0, 1006699012, 0]]], [[[128, 0, 1006699012, 0]]], [[[144, 0, 1006699012, 0]]], [[[160, 0, 1006699012, 0]]], [[[176, 0, 1006699012, 0]]], [[[192, 0, 1006699012, 0]]], [[[208, 0, 1006699012, 0]]], [[[224, 0, 1006699012, 0]]], [[[240, 0, 1006699012, 0]]]]> : tensor<16x1x1x4xsi32>>,
            weights=<dense<1.000000e+00> : tensor<16x16x1x1xf16>, [#const.Reorder<#NHWC>]>>
        ]

    return %cst : memref<1x1x1x384xf16>

    //CHECK: [[FUSED_CONSTANT:%.+]] = const.Declare memref<1x1x1x384xf16> =
    // CHECK-SAME{LITERAL}: dense<[[[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.075600e-05, 1.000980e+00, 0.000000e+00, 0.000000e+00, 9.536740e-07, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.075600e-05, 1.000980e+00, 0.000000e+00, 0.000000e+00, 1.907350e-06, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.075600e-05, 1.000980e+00, 0.000000e+00, 0.000000e+00, 2.861020e-06, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.075600e-05, 1.000980e+00, 0.000000e+00, 0.000000e+00, 3.814700e-06, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.075600e-05, 1.000980e+00, 0.000000e+00, 0.000000e+00, 4.768370e-06, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.075600e-05, 1.000980e+00, 0.000000e+00, 0.000000e+00, 5.722050e-06, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.075600e-05, 1.000980e+00, 0.000000e+00, 0.000000e+00, 6.675720e-06, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.075600e-05, 1.000980e+00, 0.000000e+00, 0.000000e+00, 7.629390e-06, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.075600e-05, 1.000980e+00, 0.000000e+00, 0.000000e+00, 8.583060e-06, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.075600e-05, 1.000980e+00, 0.000000e+00, 0.000000e+00, 9.536740e-06, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.075600e-05, 1.000980e+00, 0.000000e+00, 0.000000e+00, 1.049040e-05, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.075600e-05, 1.000980e+00, 0.000000e+00, 0.000000e+00, 1.144410e-05, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.075600e-05, 1.000980e+00, 0.000000e+00, 0.000000e+00, 1.239780e-05, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.075600e-05, 1.000980e+00, 0.000000e+00, 0.000000e+00, 1.335140e-05, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.075600e-05, 1.000980e+00, 0.000000e+00, 0.000000e+00, 1.430510e-05, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.075600e-05, 1.000980e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]]]]> : tensor<1x1x1x384xf16>
}

// -----

!qElemType = !quant.uniform<ui4:f16, 0.5>

// CHECK-LABEL: @QuantizeSubbyteSplat
func.func @QuantizeSubbyteSplat() -> memref<1x16x3x3x!qElemType> {
    %cst = const.Declare memref<1x16x3x3x!qElemType> =
        dense<5.0> : tensor<1x16x3x3xf16>,
        [
            #const.Quantize<!qElemType>
        ]

    return %cst : memref<1x16x3x3x!qElemType>

    // This will create 'dense<170> : tensor<1x1x1x72xui8>, [...]'
    // where 170 which is 0b10101010 which consists of two 0b1010 nibbles
    // and 0b1010 is decimal 10. Because the scale is 0.5 then we get 10 x 0.5 = 5.0.
    // CHECK:   [[CST:%.*]] = const.Declare memref<1x16x3x3x!qElemType> = dense<170> : tensor<1x1x1x72xui8>,
    // CHECK-SAME:  [#const.Reshape<[1, 16, 3, 3]>, #const.CastElemType<!qElemType>]
    // CHECK:   return [[CST]] : memref<1x16x3x3x!qElemType>
}

// -----

!qElemType = !quant.uniform<ui8:f16, 0.5>

// CHECK-LABEL: @QuantizeSplat
func.func @QuantizeSplat() -> memref<1x16x3x3x!qElemType> {
    %cst = const.Declare memref<1x16x3x3x!qElemType> =
        dense<5.0> : tensor<1x16x3x3xf16>,
        [
            #const.Quantize<!qElemType>
        ]

    return %cst : memref<1x16x3x3x!qElemType>

    // CHECK:   [[CST:%.*]] = const.Declare memref<1x16x3x3x!qElemType> = dense<10> : tensor<1x16x3x3xui8>
    // CHECK:   return [[CST]]
}

// -----

// CHECK-LABEL: @I1ConvertElemTypeConstFoldSplat
func.func @I1ConvertElemTypeConstFoldSplat() -> memref<1x2x3xi8> {
    %cst = const.Declare memref<1x2x3xi8> =
        dense<true> : tensor<1x2x3xi1>,
        [
            #const.ConvertElemType<i8>
        ]

    return %cst : memref<1x2x3xi8>

    // CHECK:   [[CST:%.*]] = const.Declare memref<1x2x3xi8> = dense<1> : tensor<1x2x3xi8>
    // CHECK:   return [[CST]]
}

// -----

// CHECK-LABEL: @I1ConvertElemTypeConstFoldNonSplat
func.func @I1ConvertElemTypeConstFoldNonSplat() -> memref<1x2x3x4xi8> {
    %cst = const.Declare memref<1x2x3x4xi8> =
        dense<[[[[0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 1, 1]], [[0, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]]]]> : tensor<1x2x3x4xi1>,
        [
            #const.ConvertElemType<i8>
        ]

    return %cst : memref<1x2x3x4xi8>

    // CHECK:               [[CST:%.*]] = const.Declare memref<1x2x3x4xi8>
    // CHECK-SAME{LITERAL}:     = dense<[[[[0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 1, 1]], [[0, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]]]]> : tensor<1x2x3x4xi8>
    // CHECK:               return [[CST]]
}

// -----

// CHECK-LABEL: @I4ConvertElemTypeConstFoldSplat
func.func @I4ConvertElemTypeConstFoldSplat() -> tensor<2x4xi8> {
    %cst = const.Declare tensor<2x4xi8> =
        dense_resource<blob> : tensor<2x4xi4>,
        [
            #const.ConvertElemType<i8>
        ]

    return %cst : tensor<2x4xi8>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<2x4xi8> = dense<2> : tensor<2x4xi8>
    // CHECK:   return [[CST]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x0400000022222222"
    }
  }
#-}

// -----

// CHECK-LABEL: @I4ConvertElemTypeConstFoldNonSplat
func.func @I4ConvertElemTypeConstFoldNonSplat() -> tensor<2x4xi8> {
    %cst = const.Declare tensor<2x4xi8> =
        dense_resource<blob> : tensor<2x4xi4>,
        [
            #const.ConvertElemType<i8>
        ]

    return %cst : tensor<2x4xi8>

    // CHECK:               [[CST:%.*]] = const.Declare tensor<2x4xi8>
    // CHECK-SAME{LITERAL}:     = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi8>
    // CHECK:               return [[CST]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x0400000010325476"
    }
  }
#-}

// -----

// CHECK-LABEL: @SI4ConvertElemTypeConstFoldSplat
func.func @SI4ConvertElemTypeConstFoldSplat() -> tensor<2x4xsi8> {
    %cst = const.Declare tensor<2x4xsi8> =
        dense_resource<blob> : tensor<2x4xsi4>,
        [
            #const.ConvertElemType<si8>
        ]

    return %cst : tensor<2x4xsi8>

    // CHECK:   [[CST:%.+]] = const.Declare tensor<2x4xsi8> = dense<-2> : tensor<2x4xsi8>
    // CHECK:   return [[CST]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x10000000EEEEEEEE"
    }
  }
#-}

// -----

// CHECK-LABEL: @SI4ConvertElemTypeConstFoldNonSplat
func.func @SI4ConvertElemTypeConstFoldNonSplat() -> tensor<2x4xsi8> {
    %cst = const.Declare tensor<2x4xsi8> =
        dense_resource<blob> : tensor<2x4xsi4>,
        [
            #const.ConvertElemType<si8>
        ]

    return %cst : tensor<2x4xsi8>

    // CHECK:               [[CST:%.+]] = const.Declare tensor<2x4xsi8>
    // CHECK-SAME{LITERAL}:     = dense<[[7, -8, -7, -6], [-5, -4, -3, 0]]> : tensor<2x4xsi8>
    // CHECK:               return [[CST]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x1000000087A9CB0D"
    }
  }
#-}

// -----

// CHECK-LABEL: @SI2ConvertElemTypeConstFoldSplat
func.func @SI2ConvertElemTypeConstFoldSplat() -> tensor<2x4xsi8> {
    %cst = const.Declare tensor<2x4xsi8> =
        dense_resource<blob> : tensor<2x4xsi2>,
        [
            #const.ConvertElemType<si8>
        ]

    return %cst : tensor<2x4xsi8>

    // CHECK:   [[CST:%.+]] = const.Declare tensor<2x4xsi8> = dense<-2> : tensor<2x4xsi8>
    // CHECK:   return [[CST]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x10000000AAAA"
    }
  }
#-}

// -----

// CHECK-LABEL: @SI2ConvertElemTypeConstFoldNonSplat
func.func @SI2ConvertElemTypeConstFoldNonSplat() -> tensor<2x4xsi8> {
    %cst = const.Declare tensor<2x4xsi8> =
        dense_resource<blob> : tensor<2x4xsi2>,
        [
            #const.ConvertElemType<si8>
        ]

    return %cst : tensor<2x4xsi8>

    // CHECK:               [[CST:%.+]] = const.Declare tensor<2x4xsi8>
    // CHECK-SAME{LITERAL}:     = dense<[[1, 0, -1, -2], [-1, -2, 1, 0]]> : tensor<2x4xsi8>
    // CHECK:               return [[CST]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x10000000B11B"
    }
  }
#-}

// -----

// CHECK-LABEL: @U2ConvertElemTypeConstFoldSplat
func.func @U2ConvertElemTypeConstFoldSplat() -> tensor<2x4xui8> {
    %cst = const.Declare tensor<2x4xui8> =
        dense_resource<blob> : tensor<2x4xui2>,
        [
            #const.ConvertElemType<ui8>
        ]

    return %cst : tensor<2x4xui8>

    // CHECK:   [[CST:%.+]] = const.Declare tensor<2x4xui8> = dense<2> : tensor<2x4xui8>
    // CHECK:   return [[CST]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x10000000AAAA"
    }
  }
#-}

// -----

// CHECK-LABEL: @U2ConvertElemTypeConstFoldNonSplat
func.func @U2ConvertElemTypeConstFoldNonSplat() -> tensor<2x4xui8> {
    %cst = const.Declare tensor<2x4xui8> =
        dense_resource<blob> : tensor<2x4xui2>,
        [
            #const.ConvertElemType<ui8>
        ]

    return %cst : tensor<2x4xui8>

    // CHECK:               [[CST:%.+]] = const.Declare tensor<2x4xui8>
    // CHECK-SAME{LITERAL}:     = dense<[[1, 0, 3, 2], [3, 2, 1, 0]]> : tensor<2x4xui8>
    // CHECK:               return [[CST]]
}

{-#
  dialect_resources: {
    builtin: {
      blob: "0x10000000B11B"
    }
  }
#-}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!BufferDdr = memref<40960x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>
!BufferCmx = memref<40960x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

func.func @ConstFoldWithSwizzlingSubByte(%input: !BufferDdr, %output: !BufferCmx) -> !BufferCmx {
  %bar = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
  %cst = const.Declare !BufferDdr = dense<true> : tensor<100x1x1x384xi1>, [#const.SwizzleConstant<5 : i64, 3 : i64>]
  %buf = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> !BufferCmx

  VPURT.Task waits(%bar : !VPURT.Barrier) {
    %0 = VPUIP.NNDMA inputs(%cst : !BufferDdr) outputs(%buf : !BufferCmx) -> !BufferCmx
  }

  return %buf: !BufferCmx

  // CHECK:      VPURT.DeclareVirtualBarrier
  // CHECK-DAG:      [[CST:%.+]] = const.Declare memref<40960x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}> = dense<true> : tensor<40960x1x1x1xi1>
  // CHECK-NOT:    [#const.SwizzleConstant<5 : i64, 3 : i64>]
  // CHECK:      [[BUF:%.+]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> memref<40960x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>
  // CHECK:      VPURT.Task
  // CHECK:      VPUIP.NNDMA
  // CHECK-SAME:   inputs([[CST]] : memref<40960x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>)
  // CHECK-SAME:   outputs([[BUF]] : memref<40960x1x1x1xi1, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>)
  // CHECK:      return [[BUF]]

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!BufferDdr = memref<512x1x1x1xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>
!BufferCmx = memref<512x1x1x1xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}, [@CMX_NN, 0]>

// Swizzling transformation needs to use always state of constant buffer which is an input for this transformation
func.func @ConstFoldWithSwizzlingWhereInputIsDifferentThanRawStorageValue(%input: !BufferDdr, %output: !BufferCmx) -> !BufferCmx {

  %cst = const.Declare memref<512x1x1x1xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}> = dense<1.000000e+00> : tensor<32x1x1x1xf16>, [#const.CastElemType<ui8>, #const.CastElemType<!quant.uniform<u8<0:254>:f16, 1.000000e+00>>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>>, #const.Reshape<[32, 1, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 15, 0, 0]>, #const.Reorder<affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>>, #const.SubView<[0, 0, 0, 0], [16, 16, 1, 1]>, #const.SwizzleConstant<5 : i64, 3 : i64>]

  %buf = VPURT.DeclareBuffer <CMX_NN> [0] <0> {swizzlingKey = 5 : i64} -> !BufferCmx

  VPURT.Task {
    %0 = VPUIP.NNDMA inputs(%cst : !BufferDdr) outputs(%buf : !BufferCmx) -> !BufferCmx
  }

  return %buf: !BufferCmx

  // CHECK:      const.Declare memref<512x1x1x1xui8, {order = #NHWC, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}> = dense<[
  // CHECK-COUNT-16{LITERAL}:  [[[1]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]]
  // CHECK-COUNT-16{LITERAL}:  [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]]
  // CHECK:      ]>
  // CHECK-NOT:    [#const.SwizzleConstant<5 : i64, 3 : i64>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @ConstFoldWithSwizzlingWhereContentShapeIsDifferentFromOpShape() -> memref<768x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}> {

  %cst = const.Declare memref<768x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}> = dense<[[[[1], [2], [3]]]]> : tensor<1x1x3x1xui8>, [#const.Reshape<[3, 1, 1, 1]>, #const.Broadcast<0, 768>, #const.SwizzleConstant<5 : i64, 3 : i64>]

  return %cst: memref<768x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>

  // CHECK:      [[CST:%.+]] = const.Declare memref<768x1x1x1xui8, {order = #NCHW, swizzlingScheme = #VPUIP.SwizzlingSchemeAttr<key = 5 : i64, sizeAlignment = 512 : i64>}>
  // CHECK-NOT:  [#const.SwizzleConstant<5 : i64, 3 : i64>]
}
