//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="ws-extraction-mode=gen-all" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

// This test file focuses on testing view-like-only constant transformations

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x000000040011223300aabbcc00aabbcc00aabbcc"
        }
    }
#-}

// CHECK: @NoSplit
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

    // CHECK: func.func private @init([[ARG:%.+]]: tensor<4x1xf32>)
    // CHECK:    [[CST:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>, [#const.CastElemType<f32>]
    // CHECK:    [[ADD:%.+]] = IE.Add([[ARG]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x1xf32>, tensor<1xf32> -> tensor<4x1xf32>
    // CHECK:    [[RESHAPE:%.+]] = IE.Reshape([[ADD]]) {shape_value = [2, 2]} : tensor<4x1xf32> -> tensor<2x2xf32>
    // CHECK:    [[CST0:%.+]] = const.Declare tensor<1xf32> = dense<4.200000e+01> : tensor<1xf32>, [#const.CastElemType<f32>]
    // CHECK:    IE.Multiply([[RESHAPE]], [[CST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<2x2xf32>, tensor<1xf32> -> tensor<2x2xf32>

    // CHECK:  func.func private @main([[ARG:%.+]]: tensor<2x2xf32>)
    // CHECK:    return [[ARG]]

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
// CHECK: [[CN:#.+]] = affine_map<(d0, d1) -> (d1, d0)>

// CHECK: @HalfViewLike
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

    // CHECK:  func.func private @init([[ARG:%.+]]: tensor<4x1xf32, {order = #CN}>)
    // CHECK:     [[CV:%.+]] = IE.Convert([[ARG]]) {dstElemType = f16} : tensor<4x1xf32, {order = #CN}> -> tensor<4x1xf16, {order = #CN}>
    // CHECK:     [[CST:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf32>, [#const.CastElemType<f16>]
    // CHECK:     IE.Add([[CV]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x1xf16, {order = #CN}>, tensor<1xf16> -> tensor<4x1xf16, {order = #CN}>

    // CHECK:  func.func private @main([[ARG:%.+]]: tensor<4x1xf16, {order = #CN}>)
    // CHECK:     [[CAST:%.+]] = VPU.PermuteCast([[ARG]])
    // CHECK:     VPU.Slice [[CAST]]
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
// CHECK: [[CN:#.+]] = affine_map<(d0, d1) -> (d1, d0)>

// CHECK: @MixedViewLike
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

    // CHECK:  func.func private @init([[ARG:%.+]]: tensor<4x1xf32, {order = #CN}>)
    // CHECK:     [[CV:%.+]] = IE.Convert([[ARG]]) {dstElemType = f16} : tensor<4x1xf32, {order = #CN}> -> tensor<4x1xf16, {order = #CN}>
    // CHECK:     [[CST:%.+]] = const.Declare tensor<1xf16> = dense<1.000000e+00> : tensor<1xf32>, [#const.CastElemType<f16>]
    // CHECK:     [[ADD:%.+]] = IE.Add([[CV]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x1xf16, {order = #CN}>, tensor<1xf16> -> tensor<4x1xf16, {order = #CN}>
    // CHECK:     [[REORDER:%.+]] = IE.Reorder([[ADD]]) {dstOrder = #CN} : tensor<4x1xf16, {order = #CN}> -> tensor<4x1xf16, {order = #CN}>
    // CHECK:     [[CST0:%.+]] = const.Declare tensor<1xf16> = dense<4.200000e+01> : tensor<1xf32>, [#const.CastElemType<f16>]
    // CHECK:     IE.Multiply([[REORDER]], [[CST0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x1xf16, {order = #CN}>, tensor<1xf16> -> tensor<4x1xf16, {order = #CN}>

    // CHECK:  func.func private @main([[ARG:%.+]]: tensor<4x1xf16, {order = #CN}>)
    // CHECK:     VPU.Slice [[ARG]]

}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x000000040011223300aabbcc00aabbcc00aabbcc"
        }
    }
#-}

// CHECK: @Reshape
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

    // CHECK: func.func private @init([[ARG:%.+]]: tensor<4x1xf32>)
    // CHECK:    [[CST:%.+]] = const.Declare tensor<1xf32> = dense<1.000000e+00> : tensor<1xf32>, [#const.CastElemType<f32>]
    // CHECK:    IE.Add([[ARG]], [[CST]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<4x1xf32>, tensor<1xf32> -> tensor<4x1xf32>

    // CHECK:  func.func private @main([[ARG:%.+]]: tensor<4x1xf32>)
    // CHECK:    VPU.Reshape([[ARG]])

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
// CHECK: [[CN:#.+]] = affine_map<(d0, d1) -> (d1, d0)>

// CHECK: @LayoutCast
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

    // CHECK: @main([[ARG:%.+]]: tensor<4x1xf32>) -> tensor<4x1xf32, {order = [[CN]]}>
    // CHECK:   VPU.LayoutCast([[ARG]])
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
// CHECK-DAG: [[NC:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: [[CN:#.+]] = affine_map<(d0, d1) -> (d1, d0)>

// CHECK: @TrivialMemPermute
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

    // CHECK: @main([[ARG:%.+]]: tensor<4x1xf32, {order = #CN}>) -> tensor<4x1xf32>
    // CHECK:   VPU.PermuteCast([[ARG]])
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

// CHECK: @TrivialTranspose
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

    // CHECK: @main([[ARG:%.+]]: tensor<4x1xf32>) -> tensor<1x4xf32>
    // CHECK:   VPU.ShapeCast {shape = [1, 4]} inputs([[ARG]]
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

// CHECK: module @AffineReshape
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

    // CHECK: func.func private @main([[ARG:%.+]]: tensor<1x1x3x3xf32>)
    // CHECK:      VPU.AffineReshape([[ARG]])
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
// CHECK: [[CN:#.+]] = affine_map<(d0, d1) -> (d1, d0)>

// CHECK: @ReshapeNonIdentity
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

    // CHECK:  func.func private @main([[ARG:%.+]]: tensor<4x1xf32, {order = #CN}>)
    // CHECK:    VPU.ShapeCast {shape = [2, 2]} inputs([[ARG]]

}
