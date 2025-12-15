//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --introduce-init-function="ws-extraction-mode=gen-init init-part=0 memory-limit=1000" --verify-diagnostics %s | FileCheck --check-prefix=CHECK-BIG-LIMIT %s
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --introduce-init-function="ws-extraction-mode=gen-init init-part=0 memory-limit=4" --verify-diagnostics %s | FileCheck --check-prefix=CHECK-SMALL-LIMIT %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

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
    // CHECK-SMALL-LIMIT-NEXT:   [[ONE:%.+]] = const.Declare {{.*}} dense<1.000000e+00>
    // CHECK-SMALL-LIMIT-NEXT:   [[ADD_ONE:%.+]] = IE.Add([[OV_1]], [[ONE]])
    // CHECK-SMALL-LIMIT-NEXT:   return [[ADD_ONE]]

    // CHECK-SMALL-LIMIT-NOT: func.func @init_part1
}
