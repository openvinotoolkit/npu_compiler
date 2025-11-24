//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="ws-extraction-mode=gen-init" %s | FileCheck --check-prefix=CHECK-INIT %s
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="ws-extraction-mode=gen-main" %s | FileCheck --check-prefix=CHECK-MAIN %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// This test file focuses on testing view-like-only constant transformations

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x000000040011223300aabbcc00aabbcc00aabbcc"
        }
    }
#-}

// CHECK-INIT: @NoSplit
// CHECK-MAIN: @NoSplit
module @NoSplit {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x2xf32>
    }

    func.func @main() -> tensor<2x2xf32> {
        %cst = const.Declare tensor<2x2xf32> = dense_resource<vpux_ow_1> : tensor<4x1xf32>,
            [#const.Add<1.0>, #const.Reshape<[2, 2]>, #const.Rescale<42.0>]
        return %cst : tensor<2x2xf32>
    }

    // CHECK-INIT:  func.func @init([[ARG:%.+]]: tensor<4x1xf32>)
    // CHECK-INIT:      [[CST:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>, [#const.CastElemType<f32>]
    // CHECK-INIT:      [[ADD:%.+]] = IE.Add([[ARG]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x1xf32>, tensor<1xf32> -> tensor<4x1xf32>
    // CHECK-INIT:      [[RESHAPE:%.+]] = IE.Reshape([[ADD]]) {shape_value = [2, 2]} : tensor<4x1xf32> -> tensor<2x2xf32>
    // CHECK-INIT:      [[CST0:%.+]] = const.Declare tensor<1xf32> = dense<4.200000e+01> : tensor<1xf32>, [#const.CastElemType<f32>]
    // CHECK-INIT:      IE.Multiply([[RESHAPE]], [[CST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x2xf32>, tensor<1xf32> -> tensor<2x2xf32>

    // CHECK-MAIN:  func.func @main([[ARG:%.+]]: tensor<2x2xf32>)
    // CHECK-MAIN:      return [[ARG]]

}


// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x000000040011223300aabbcc00aabbcc00aabbcc"
        }
    }
#-}

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-INIT: @HalfViewLike
// CHECK-MAIN: @HalfViewLike
module @HalfViewLike {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x1xf16, {order = #CN}>
    }

    func.func @main() -> tensor<2x1xf16, {order = #CN}> {
        %cst = const.Declare tensor<2x1xf16, {order = #CN}> = dense_resource<vpux_ow_1> : tensor<4x1xf32, {order = #CN}>,
            [#const.CastElemType<f16>, #const.Add<1.0>, #const.Reorder<#CN>, #const.SubView<[0, 0], [2, 1]>]
        return %cst : tensor<2x1xf16, {order = #CN}>
    }

    // CHECK-INIT:  func.func @init([[ARG:%.+]]: tensor<4x1xf32, {order = #CN}>)
    // CHECK-INIT:      [[CV:%.+]] = IE.Convert([[ARG]]) {dstElemType = f16} : tensor<4x1xf32, {order = #CN}> -> tensor<4x1xf16, {order = #CN}>
    // CHECK-INIT:      [[CST:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf32>, [#const.CastElemType<f16>]
    // CHECK-INIT:      IE.Add([[CV]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}

    // CHECK-MAIN:  func.func @main([[ARG:%.+]]: tensor<4x1xf16, {order = #CN}>)
    // CHECK-MAIN:      [[CAST:%.+]] = VPU.PermuteCast([[ARG]])
    // CHECK-MAIN:      VPU.Slice [[CAST]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x000000040011223300aabbcc00aabbcc00aabbcc"
        }
    }
#-}

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-INIT: @MixedViewLike
// CHECK-MAIN: @MixedViewLike
module @MixedViewLike {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x1xf16, {order = #CN}>
    }

    func.func @main() -> tensor<2x1xf16, {order = #CN}> {
        %cst = const.Declare tensor<2x1xf16, {order = #CN}> = dense_resource<vpux_ow_1> : tensor<4x1xf32, {order = #CN}>,
            [#const.CastElemType<f16>, #const.Add<1.0>, #const.Reorder<#CN>, #const.Rescale<42.0>, #const.SubView<[0, 0], [2, 1]>]
        return %cst : tensor<2x1xf16, {order = #CN}>
    }

    // CHECK-INIT:  func.func @init([[ARG:%.+]]: tensor<4x1xf32, {order = #CN}>)
    // CHECK-INIT:      [[CV:%.+]] = IE.Convert([[ARG]]) {dstElemType = f16} : tensor<4x1xf32, {order = #CN}> -> tensor<4x1xf16, {order = #CN}>
    // CHECK-INIT:      [[CST:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf32>, [#const.CastElemType<f16>]
    // CHECK-INIT:      [[ADD:%.+]] = IE.Add([[CV]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
    // CHECK-INIT:      [[REORDER:%.+]] = IE.Reorder([[ADD]]) {dstOrder = #CN} : tensor<4x1xf16, {order = #CN}> -> tensor<4x1xf16, {order = #CN}>
    // CHECK-INIT:      [[CST0:%.+]] = const.Declare tensor<1xf16> = dense<4.200000e+01> : tensor<1xf32>, [#const.CastElemType<f16>]
    // CHECK-INIT:      IE.Multiply([[REORDER]], [[CST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}

    // CHECK-MAIN:  func.func @main([[ARG:%.+]]: tensor<4x1xf16, {order = #CN}>)
    // CHECK-MAIN:      VPU.Slice [[ARG]]

}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x000000040011223300aabbcc00aabbcc00aabbcc"
        }
    }
#-}

// CHECK-INIT: @Reshape
// CHECK-MAIN: @Reshape
module @Reshape {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x2xf32>
    }

    func.func @main() -> tensor<2x2xf32> {
        %cst = const.Declare tensor<2x2xf32> = dense_resource<vpux_ow_1> : tensor<4x1xf32>,
            [#const.Add<1.0>, #const.Reshape<[2, 2]>]
        return %cst : tensor<2x2xf32>
    }

    // CHECK-INIT:  func.func @init([[ARG:%.+]]: tensor<4x1xf32>)
    // CHECK-INIT:      [[CST:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>, [#const.CastElemType<f32>]
    // CHECK-INIT:      IE.Add([[ARG]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x1xf32>, tensor<1xf32> -> tensor<4x1xf32>

    // CHECK-MAIN:  func.func @main([[ARG:%.+]]: tensor<4x1xf32>)
    // CHECK-MAIN:      VPU.Reshape([[ARG]])

}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x000000040011223300aabbcc00aabbcc00aabbcc"
        }
    }
#-}

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-INIT: @LayoutCast
// CHECK-MAIN: @LayoutCast
module @LayoutCast {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<4x1xf32, {order = #CN}>
    }

    func.func @main() -> tensor<4x1xf32, {order = #CN}> {
        %cst = const.Declare tensor<4x1xf32, {order = #CN}> = dense_resource<vpux_ow_1> : tensor<4x1xf32>,
            [#const.Add<1.0>, #const.LayoutCast<#CN>]
        return %cst : tensor<4x1xf32, {order = #CN}>
    }

    // CHECK-MAIN:  @main([[ARG:%.+]]: tensor<4x1xf32>) -> tensor<4x1xf32, {order = #CN}>
    // CHECK-MAIN:      VPU.LayoutCast([[ARG]])
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x000000040011223300aabbcc00aabbcc00aabbcc"
        }
    }
#-}

#NC = affine_map<(d0, d1) -> (d0, d1)>
#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-INIT: @TrivialMemPermute
// CHECK-MAIN: @TrivialMemPermute
module @TrivialMemPermute {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<4x1xf32>
    }

    func.func @main() -> tensor<4x1xf32> {
        %cst = const.Declare tensor<4x1xf32> = dense_resource<vpux_ow_1> : tensor<4x1xf32, {order = #CN}>,
            [#const.Add<1.0>, #const.MemPermute<#NC, #CN>]
        return %cst : tensor<4x1xf32>
    }

    // CHECK-MAIN:  @main([[ARG:%.+]]: tensor<4x1xf32, {order = #CN}>) -> tensor<4x1xf32>
    // CHECK-MAIN:      VPU.PermuteCast([[ARG]])
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x000000040011223300aabbcc00aabbcc00aabbcc"
        }
    }
#-}

// Note: it is trivial in combination with the type:
#swap = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-INIT: @TrivialTranspose
// CHECK-MAIN: @TrivialTranspose
module @TrivialTranspose {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<1x4xf32>
    }

    func.func @main() -> tensor<1x4xf32> {
        %cst = const.Declare tensor<1x4xf32> = dense_resource<vpux_ow_1> : tensor<4x1xf32>,
            [#const.Add<1.0>, #const.Transpose<#swap>]
        return %cst : tensor<1x4xf32>
    }

    // CHECK-MAIN:  @main([[ARG:%.+]]: tensor<4x1xf32>) -> tensor<1x4xf32>
    // CHECK-MAIN:      VPU.ShapeCast {shape = [1, 4]} inputs([[ARG]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x00000004aabbccddaabbccddaabbccddaabbccddaabbccddaabbccddaabbccddaabbccddaabbccd6"
        }
    }
#-}

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-INIT: module @AffineReshape
// CHECK-MAIN: module @AffineReshape
module @AffineReshape {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<1x1x3x3xf32>
    }

    func.func @main() -> tensor<1x1x3x3xf32, {order = #NCWH}> {
        %cst = const.Declare tensor<1x1x3x3xf32, {order = #NCWH}> = dense_resource<vpux_ow_1> : tensor<1x1x3x3xf32>,
            [#const.Add<1.0>, #const.AffineReshape<[[0], [1], [3], [2]], [1, 1, 3, 3]>]
        return %cst : tensor<1x1x3x3xf32, {order = #NCWH}>
    }

    // CHECK-MAIN:  func.func @main([[ARG:%.+]]: tensor<1x1x3x3xf32>)
    // CHECK-MAIN:      VPU.AffineReshape([[ARG]])
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x000000040011223300aabbcc00aabbcc00aabbcc"
        }
    }
#-}

#CN = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-INIT: @ReshapeNonIdentity
// CHECK-MAIN: @ReshapeNonIdentity
module @ReshapeNonIdentity {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x2xf32, {order = #CN}>
    }

    func.func @main() -> tensor<2x2xf32, {order = #CN}> {
        %cst = const.Declare tensor<2x2xf32, {order = #CN}> = dense_resource<vpux_ow_1> : tensor<4x1xf32, {order = #CN}>,
            [#const.Add<1.0>, #const.Reshape<[2, 2]>]
        return %cst : tensor<2x2xf32, {order = #CN}>
    }

    // CHECK-MAIN:  func.func @main([[ARG:%.+]]: tensor<4x1xf32, {order = #CN}>)
    // CHECK-MAIN:      VPU.ShapeCast {shape = [2, 2]} inputs([[ARG]]
}
