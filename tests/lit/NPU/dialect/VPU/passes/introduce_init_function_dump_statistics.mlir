//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: env OV_NPU_LOG_LEVEL=LOG_INFO vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --introduce-init-function="ws-extraction-mode=gen-init" -o /dev/null %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX


{-#
    dialect_resources: {
        builtin: {
            vpux_ow_11bytes: "0x000000040011223300aabbcc00aabb",
            vpux_ow_10bytes: "0x00000004aabbccddee1122334455"
        }
    }
#-}

module @SizePreserved {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1xf32>
    } outputsInfo : {
        DataInfo "output1" : tensor<1xf32>
    }

    func.func @main_part0(%arg: tensor<1xf32>) -> tensor<1xf32> {
        %cst_ov_11bytes = const.Declare tensor<11xui8> = dense_resource<vpux_ow_11bytes> : tensor<11xui8>,
            [#const.CastElemType<f16>, #const.Add<42.0>, #const.CastElemType<ui8>]
        return %arg : tensor<1xf32>
    }

    func.func @main(%arg: tensor<1xf32>) -> tensor<1xf32> {
        %cst_ov_10bytes = const.Declare tensor<5xf16> = dense_resource<vpux_ow_10bytes> : tensor<5xf16>,
            [#const.Add<42.0>]

        %call = func.call @main_part0(%arg) : (tensor<1xf32>) -> tensor<1xf32>
        return %call : tensor<1xf32>
    }

    // Note: total bytes = 10 + 11 = 21 ~ 0.02 KB (21 / 1024)

    // CHECK:   Summary about constants:
    // CHECK:    All imported unique weights: 2 (0.02 KB)
    // CHECK:    Available unique weights[1]: 2 (0.02 KB which is 100.00%)
    // CHECK:    Unique weights used by schedule (from available): 2 (0.02 KB which is 100.00%)
    // CHECK:    OV-originated constants[2] in IR: 2 (0.02 KB)
    // CHECK:    Unused constants[3]: 0 (0.00 KB which is 0.00%)
    // CHECK:    Unsupported constants[4]: 0 (0.00 KB which is 0.00%)
    // CHECK:    Size percentage of *used* constants: 100.00%
    // CHECK:    Generated schedule's total I/O size: 0.04 KB
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_2bytes: "0x00000004aabb"
        }
    }
#-}

module @LargeOutput {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1xf32>
    } outputsInfo : {
        DataInfo "output1" : tensor<1xf32>
    }

    func.func @main(%arg: tensor<1xf32>) -> tensor<1xf32> {
        %cst_ov_2bytes = const.Declare tensor<1024xsi8> = dense_resource<vpux_ow_2bytes> : tensor<2xsi8>,
            [#const.PadWithZero<[0], [1022]>]

        return %arg: tensor<1xf32>
    }

    // CHECK:   Summary about constants:
    // CHECK:    All imported unique weights: 1 (0.00 KB)
    // CHECK:    Available unique weights[1]: 1 (0.00 KB which is 100.00%)
    // CHECK:    Unique weights used by schedule (from available): 1 (0.00 KB which is 100.00%)
    // CHECK:    OV-originated constants[2] in IR: 1 (1.00 KB)
    // CHECK:    Unused constants[3]: 0 (0.00 KB which is 0.00%)
    // CHECK:    Unsupported constants[4]: 0 (0.00 KB which is 0.00%)
    // CHECK:    Size percentage of *used* constants: 100.00%
    // CHECK:    Generated schedule's total I/O size: 1.00 KB
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x000000040011223300aabbcc00aabbcc00aabbcc",
            vpux_ow_2: "0x0000000400112233",
            vpux_ow_outlined: "0x00000004aabbccddee",
            vpux_ow_splat: "0x0000000412341234",
            vpux_ow_noop: "0x000000040011223300aabbcc00aabbcc00aabbcc",

            vpux_ow_unused: "0x0000000400112233"
        }
    }
#-}

!qElemType1 = !quant.uniform<i8:f16, 0.5>
!qElemType2 = !quant.uniform<ui8:f16, 0.5:128>

module @ManyConstants {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1xf32>
    } outputsInfo : {
        DataInfo "output1" : tensor<1xf32>
    }

    func.func @main_part0(%arg: tensor<1xf32>) -> tensor<1xf32> {
        %cst_ov_outlined = const.Declare tensor<5xf16> = dense_resource<vpux_ow_outlined> : tensor<5xui8>,
            [#const.CastElemType<f16>, #const.Add<42.0>]

        // Not suitable for weights separation below:
        %cst_ov_splat = const.Declare tensor<2xf16> = dense_resource<vpux_ow_splat> : tensor<2xf16>,
            [#const.Add<1.0>]
        %cst_ov_noop = const.Declare tensor<2x2xf32> = dense_resource<vpux_ow_noop> : tensor<2x2xf32>
        %cst_ov1_non_supported = const.Declare tensor<1x1x3x3xf32> = dense_resource<vpux_ow_1> : tensor<1x1x2x2xf32>,
            [#const.ExpandDilated<[2, 2]>]

        return %arg : tensor<1xf32>
    }

    func.func @main(%arg: tensor<1xf32>) -> tensor<1xf32> {
        %cst_ov1_0 = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_1> : tensor<2x2xf32>,
            [#const.CastElemType<f16>]
        %cst_ov1_1 = const.Declare tensor<2x2xf32> = dense_resource<vpux_ow_1> : tensor<2x2xf32>,
            [#const.Add<1.0>]

        %cst_ov2 = const.Declare tensor<2x!qElemType1> = dense_resource<vpux_ow_2> : tensor<2xf16>,
            [#const.Rescale<0.5>, #const.CastElemType<!qElemType1>]


        %not_ov_weight = const.Declare tensor<2x2xf16> = dense<[[4.0, 2.0], [12.0, 18.0]]> : tensor<2x2xf16>,
            [#const.Add<42.0>]


        %call = func.call @main_part0(%arg) : (tensor<1xf32>) -> tensor<1xf32>
        return %call : tensor<1xf32>
    }

    // CHECK:   Summary about constants:
    // CHECK:    All imported unique weights: 6 (0.05 KB)
    // CHECK:    Available unique weights[1]: 5 (0.04 KB which is 91.84%)
    // CHECK:    Unique weights used by schedule (from available): 3 (0.02 KB which is 55.56%)
    // CHECK:    OV-originated constants[2] in IR: 7 (0.09 KB)
    // CHECK:    Unused constants[3]: 2 (0.02 KB which is 21.74%)
    // CHECK:    Unsupported constants[4]: 1 (0.04 KB which is 39.13%)
    // CHECK:    Size percentage of *used* constants: 39.13%
    // CHECK:    Generated schedule's total I/O size: 0.06 KB

    // CHECK: [1]: available unique weights - weights that come from original model and are used in the compiled schedule (via constant operations)
    // CHECK: [2]: OV-originated constants - constant operations that combine OV weights with transformations (e.g. subview, reorder)
    // CHECK:  Note: the same unique weight could be used in multiple constants
    // CHECK: [3]: unused constants - OV-originated constants that are ignored by weights separation (e.g. splats, only trivial transformations)
    // CHECK: [4]: unsupported constants - OV-originated constants that have unsupported transformations
}
