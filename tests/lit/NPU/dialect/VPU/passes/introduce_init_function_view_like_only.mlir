//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="ws-extraction-mode=gen-main" %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// This test file focuses on testing view-like-only constant transformations

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x000000040011223300aabbcc00aabbcc00aabbcc"
        }
    }
#-}

// CHECK: @NoTransformations
module @NoTransformations {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<4x1xf32>
    }

    func.func @main() -> tensor<4x1xf32> {
        %cst = const.Declare tensor<4x1xf32> = dense_resource<vpux_ow_1> : tensor<4x1xf32>
        return %cst : tensor<4x1xf32>
    }

    // CHECK: @main() -> tensor<4x1xf32>
    // CHECK:   [[CST:%.+]] = const.Declare {{.*}} dense_resource<vpux_ow_1>
    // CHECK:   return [[CST]]
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
            [#const.Reshape<[2, 2]>]
        return %cst : tensor<2x2xf32>
    }

    // CHECK: @main() -> tensor<2x2xf32>
    // CHECK:   [[CST:%.+]] = const.Declare {{.*}} dense_resource<vpux_ow_1> {{.*}} [#const.Reshape<[2, 2]>]
    // CHECK:   return [[CST]]
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

// CHECK: @ReshapeNonIdentityOrder
module @ReshapeNonIdentityOrder {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x2xf32, {order = #CN}>
    }

    func.func @main() -> tensor<2x2xf32, {order = #CN}> {
        %cst = const.Declare tensor<2x2xf32, {order = #CN}> = dense_resource<vpux_ow_1> : tensor<4x1xf32, {order = #CN}>,
            [#const.Reshape<[2, 2]>]
        return %cst : tensor<2x2xf32, {order = #CN}>
    }

    // CHECK: @main() -> tensor<2x2xf32, {order = [[CN]]}>
    // CHECK:   [[CST:%.+]] = const.Declare {{.*}} dense_resource<vpux_ow_1> {{.*}} [#const.Reshape<[2, 2]>]
    // CHECK:   return [[CST]]
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x000000040011223300aabbcc00aabbcc00aabbcc"
        }
    }
#-}

// CHECK: @SubView
module @SubView {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<2x1xf32>
    }

    func.func @main() -> tensor<2x1xf32> {
        %cst = const.Declare tensor<2x1xf32> = dense_resource<vpux_ow_1> : tensor<4x1xf32>,
            [#const.SubView<[0, 0], [2, 1]>]
        return %cst : tensor<2x1xf32>
    }

    // CHECK: @main() -> tensor<2x1xf32>
    // CHECK:   [[CST:%.+]] = const.Declare {{.*}} dense_resource<vpux_ow_1> {{.*}} [#const.SubView<[0, 0], [2, 1]>]
    // CHECK:   return [[CST]]
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
            [#const.LayoutCast<#CN>]
        return %cst : tensor<4x1xf32, {order = #CN}>
    }

    // CHECK: @main() -> tensor<4x1xf32, {order = [[CN]]}>
    // CHECK:   [[CST:%.+]] = const.Declare {{.*}} dense_resource<vpux_ow_1> {{.*}} [#const.LayoutCast<[[CN]]>]
    // CHECK:   return [[CST]]
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
            [#const.MemPermute<#NC, #CN>]
        return %cst : tensor<4x1xf32>
    }

    // CHECK: @main() -> tensor<4x1xf32>
    // CHECK:   [[CST:%.+]] = const.Declare {{.*}} dense_resource<vpux_ow_1> {{.*}} [#const.MemPermute<[[NC]], [[CN]]>]
    // CHECK:   return [[CST]]
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
// CHECK: [[swap:#.+]] = affine_map<(d0, d1) -> (d1, d0)>

// CHECK: @TrivialTranspose
module @TrivialTranspose {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<1x4xf32>
    }

    func.func @main() -> tensor<1x4xf32> {
        %cst = const.Declare tensor<1x4xf32> = dense_resource<vpux_ow_1> : tensor<4x1xf32>,
            [#const.Transpose<#swap>]
        return %cst : tensor<1x4xf32>
    }

    // CHECK: @main() -> tensor<1x4xf32>
    // CHECK:   [[CST:%.+]] = const.Declare {{.*}} dense_resource<vpux_ow_1> {{.*}} [#const.Transpose<[[swap]]>]
    // CHECK:   return [[CST]]
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

// CHECK: @TrivialReorder
module @TrivialReorder {
    net.NetworkInfo entryPoint : @main inputsInfo : {
    } outputsInfo : {
        DataInfo "output1" : tensor<4x1xf32, {order = #CN}>
    }

    func.func @main() -> tensor<4x1xf32, {order = #CN}> {
        %cst = const.Declare tensor<4x1xf32, {order = #CN}> = dense_resource<vpux_ow_1> : tensor<4x1xf32>,
            [#const.Reorder<#CN>]
        return %cst : tensor<4x1xf32, {order = #CN}>
    }

    // CHECK: @main() -> tensor<4x1xf32, {order = [[CN]]}>
    // CHECK:   [[CST:%.+]] = const.Declare {{.*}} dense_resource<vpux_ow_1> {{.*}} [#const.Reorder<[[CN]]>]
    // CHECK:   return [[CST]]
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
            [#const.AffineReshape<[[0], [1], [3], [2]], [1, 1, 3, 3]>]
        return %cst : tensor<1x1x3x3xf32, {order = #NCWH}>
    }

    // CHECK:           func.func @main() -> tensor<1x1x3x3xf32, {order = #NCWH}>
    // CHECK:               [[CST:%.+]] = const.Declare {{.*}} dense_resource<vpux_ow_1>
    // CHECK{LITERAL}:           [#const.AffineReshape<[[0], [1], [3], [2]], [1, 1, 3, 3]>]
    // CHECK:           return [[CST]]
}
