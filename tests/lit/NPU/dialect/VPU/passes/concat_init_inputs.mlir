//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler=vpu-arch=%arch% --concat-init-inputs %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// CHECK: module @SingleInput
module @SingleInput {
    net.NetworkInfo entryPoint : @init inputsInfo : {
        DataInfo "vpux_ow_1" : tensor<2x2x5x3xf16>
    } outputsInfo : {
        DataInfo "vpux_tw_1_hash_123456789" : tensor<2x2x5x3xf16>
    }

    // CHECK: inputsInfo
    // CHECK-NEXT: DataInfo "vpux_ow_1" : tensor<2x2x5x3xf16>
    // CHECK: outputsInfo
    // CHECK-NEXT: DataInfo "vpux_tw_1_hash_123456789" : tensor<2x2x5x3xf16>

    func.func @init(%ov1: tensor<2x2x5x3xf16>) -> tensor<2x2x5x3xf16> {
        %one = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
        %res = IE.Add(%ov1, %one) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<2x2x5x3xf16>, tensor<1xf16> -> tensor<2x2x5x3xf16>
        return %res : tensor<2x2x5x3xf16>
    }

    // CHECK: func.func @init([[OV1:%.+]]: tensor<2x2x5x3xf16>) -> tensor<2x2x5x3xf16>
    // CHECK:   [[RES:%.+]] = IE.Add([[OV1]], {{%.+}})
    // CHECK:   return [[RES]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: module @TwoInputs
module @TwoInputs {
    net.NetworkInfo entryPoint : @init inputsInfo : {
        DataInfo "vpux_ow_1" : tensor<2x2x5x3xf16>
        DataInfo "vpux_ow_2" : tensor<42x100x1x1xui8>
    } outputsInfo : {
        DataInfo "vpux_tw_1_hash_123456789" : tensor<2x2x5x3xf16>
        DataInfo "vpux_tw_2_hash_987654321" : tensor<42x100x1x1xsi8>
    }

    // CHECK: inputsInfo
    // CHECK-NEXT: DataInfo "vpux_ow_hash_2211455849133395826_concat" : tensor<4320xi8>
    // CHECK: outputsInfo
    // CHECK-NEXT: DataInfo "vpux_tw_1_hash_123456789" : tensor<2x2x5x3xf16>
    // CHECK-NEXT: DataInfo "vpux_tw_2_hash_987654321" : tensor<42x100x1x1xsi8>

    func.func @init(%ov1: tensor<2x2x5x3xf16>, %ov2: tensor<42x100x1x1xui8, {order = #NHWC}>)
            -> (tensor<2x2x5x3xf16>, tensor<42x100x1x1xsi8, {order = #NHWC}>) {
        %one = const.Declare tensor<1xf16> = dense<1.0> : tensor<1xf16>
        %res1 = IE.Add(%ov1, %one) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>}
            : tensor<2x2x5x3xf16>, tensor<1xf16> -> tensor<2x2x5x3xf16>

        %res2 = IE.Convert(%ov2) {dstElemType = si8}
            : tensor<42x100x1x1xui8, {order = #NHWC}> -> tensor<42x100x1x1xsi8, {order = #NHWC}>

        return %res1, %res2 : tensor<2x2x5x3xf16>, tensor<42x100x1x1xsi8, {order = #NHWC}>
    }

    // CHECK: func.func @init([[BLOB:%.+]]: tensor<4320xi8>)
    // CHECK-SAME:  -> (tensor<2x2x5x3xf16>, tensor<42x100x1x1xsi8, {order = #NHWC}>)
    // CHECK:   [[SLICE_OV1:%.+]] = IE.Slice [[BLOB]] [0] [120]
    // CHECK:   [[RESTORED_OV1:%.+]] = Core.ReinterpretCast([[SLICE_OV1]]) {{.*}} -> tensor<2x2x5x3xf16>
    // CHECK:   [[SLICE_OV2:%.+]] = IE.Slice [[BLOB]] [120] [4200]
    // CHECK:   [[RESTORED_OV2:%.+]] = Core.ReinterpretCast([[SLICE_OV2]])
    // CHECK-SAME:  -> tensor<42x100x1x1xui8, {order = #NHWC}>

    // CHECK:   [[RES1:%.+]] = IE.Add([[RESTORED_OV1]], {{%.+}})
    // CHECK:   [[RES2:%.+]] = IE.Convert([[RESTORED_OV2]])
    // CHECK:   return [[RES1]], [[RES2]]
}
