//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --construct-ws-analysis --introduce-init-function="ws-extraction-mode=gen-init init-part=0 memory-limit=1000" --verify-diagnostics %s | FileCheck --check-prefix=CHECK-BIG-LIMIT %s
// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --construct-ws-analysis --introduce-init-function="ws-extraction-mode=gen-init init-part=0 memory-limit=4" --verify-diagnostics %s | FileCheck --check-prefix=CHECK-SMALL-LIMIT %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

// Note: these tests verify init schedule slicing. They are not supposed to test
// everything but rather test the bare minimum, focusing on the slicing logic.

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000AABBCCDD",
            vpux_ow_2: "0x10000000AABBCCDD"
        }
    }
#-}

// CHECK-BIG-LIMIT: module @MemoryLimitTest
// CHECK-SMALL-LIMIT: module @MemoryLimitTest
module @MemoryLimitTest {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<4x16xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<4x16xf16>
    }

    func.func @main(%arg: tensor<4x16xf16>) -> tensor<4x16xf16> {
        %cst1 = const.Declare tensor<4xui8> = dense_resource<vpux_ow_1> : tensor<4xui8>, [#const.Add<1.0>]
        %cst2 = const.Declare tensor<4xui8> = dense_resource<vpux_ow_2> : tensor<4xui8>, [#const.Add<2.0>]
        return %arg : tensor<4x16xf16>
    }

    // Note: large limit results in single init function being present

    // CHECK-BIG-LIMIT: func.func @init([[OV_1:%.+]]: tensor<4xui8>, [[OV_2:%.+]]: tensor<4xui8>)


    // Note: small limit results in only init part0 being present (another init
    //       part is not here because we only generate single-part schedule)

    // CHECK-SMALL-LIMIT: func.func @init_part0([[OV_1:%.+]]: tensor<4xui8>)
    // CHECK-SMALL-LIMIT-NEXT:   [[ONE:%.+]] = const.Declare {{.+}} dense<1.000000e+00>
    // CHECK-SMALL-LIMIT-NEXT:   [[ADD_ONE:%.+]] = IE.Add([[OV_1]], [[ONE]])
    // CHECK-SMALL-LIMIT-NEXT:   return [[ADD_ONE]]

    // CHECK-SMALL-LIMIT-NOT: func.func @init_part1
}

// -----

{-#
  dialect_resources: {
    builtin: {
            vpux_ow_0: "0x10000000ABCDABCDABCDABCE"
        }
  }
#-}

// Note: this test verifies that same-blob constants remain together (in the
// same init schedule), even when their type or shape do not match e.g.
// `dense_resource<blob> : tensor<2x3x4xf16>` and `dense_resource<blob> : tensor<24xf32>`.
// This is possible due to OV model compression feature that can squash same-binary-data
// constants and make multiple such constants point to the same buffer.

// CHECK-BIG-LIMIT: @SameBlobConstants
// CHECK-SMALL-LIMIT: @SameBlobConstants
module @SameBlobConstants {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<2x2xf16>
    } outputsInfo : {
        DataInfo "output" : tensor<2x2xf16>
    }

    // Note: large limit results in single init function being present

    // CHECK-BIG-LIMIT: func.func @init([[ORIG:%.+]]: tensor<2x2xf16>, [[NEWSHAPE:%.+]]: tensor<4xf16>, [[NEWTYPE:%.+]]: tensor<2x2xi16>)


    // Note: small limit still results in single init function, because
    //       same-blob constants must not be split (due to compiler-plugin contract)

    // CHECK-SMALL-LIMIT: func.func @init([[ORIG:%.+]]: tensor<2x2xf16>, [[NEWSHAPE:%.+]]: tensor<4xf16>, [[NEWTYPE:%.+]]: tensor<2x2xi16>)

    func.func @main(%dummy: tensor<2x2xf16>) -> tensor<2x2xf16> {
        %orig = const.Declare tensor<2x2xf16> = dense_resource<vpux_ow_0> : tensor<2x2xf16>,
            [#const.Add<42.0>]
        %newshape = const.Declare tensor<3x2xf16> = dense_resource<vpux_ow_0> : tensor<4xf16>,
            [#const.Rescale<2.0>, #const.Reshape<[2, 2]>, #const.PadWithZero<[0, 0], [1, 0]>]
        %newtype = const.Declare tensor<2x3xf16> = dense_resource<vpux_ow_0> : tensor<2x2xi16>,
            [#const.CastElemType<f16>, #const.PadWithZero<[0, 0], [0, 1]>]
        return %dummy : tensor<2x2xf16>
    }
}
