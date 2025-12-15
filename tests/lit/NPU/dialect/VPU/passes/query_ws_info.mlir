//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --query-ws-info %s | FileCheck --check-prefix=CHECK-DEFAULT %s
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --query-ws-info="memory-limit=1000" %s | FileCheck --check-prefix=CHECK-BIG-LIMIT %s
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --query-ws-info="memory-limit=4" %s | FileCheck --check-prefix=CHECK-SMALL-LIMIT %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK-DEFAULT: module @NoConstants attributes {{.*}}VPU.WsTotalInitPartCount = 0 : i64
// CHECK-BIG-LIMIT: module @NoConstants attributes {{.*}}VPU.WsTotalInitPartCount = 0 : i64
// CHECK-SMALL-LIMIT: module @NoConstants attributes {{.*}}VPU.WsTotalInitPartCount = 0 : i64
module @NoConstants {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<4x16xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<4x16xf16>
    }

    func.func @main(%arg: tensor<4x16xf16>) -> tensor<4x16xf16> {
        %cst1 = const.Declare tensor<4xui8> = dense<42> : tensor<4xui8>, [#const.Add<1.0>]
        return %arg : tensor<4x16xf16>
    }
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000AABBCCDD",
            vpux_ow_2: "0x10000000AABBCCDD"
        }
    }
#-}

// CHECK-DEFAULT: module @WsInfo attributes {{.*}}VPU.WsTotalInitPartCount = 1 : i64
// CHECK-BIG-LIMIT: module @WsInfo attributes {{.*}}VPU.WsTotalInitPartCount = 1 : i64
// CHECK-SMALL-LIMIT: module @WsInfo attributes {{.*}}VPU.WsTotalInitPartCount = 2 : i64
module @WsInfo {
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
}
