//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler=vpu-arch=%arch% --introduce-init-function="ws-extraction-mode=gen-init" --concat-init-results="ws-extraction-mode=gen-init" %s | FileCheck --check-prefix=CHECK-INIT-FULL %s
// RUN: vpux-opt --split-input-file --init-compiler=vpu-arch=%arch% --introduce-init-function="ws-extraction-mode=gen-main" --concat-init-results="ws-extraction-mode=gen-main" %s | FileCheck --check-prefix=CHECK-MAIN-FULL %s
// RUN: vpux-opt --split-input-file --init-compiler=vpu-arch=%arch% --introduce-init-function="ws-extraction-mode=gen-init memory-limit=0 init-part=0" --concat-init-results="ws-extraction-mode=gen-init memory-limit=0 init-part=0" %s | FileCheck --check-prefix=CHECK-INIT-PART0 %s
// RUN: vpux-opt --split-input-file --init-compiler=vpu-arch=%arch% --introduce-init-function="ws-extraction-mode=gen-init memory-limit=0 init-part=1" --concat-init-results="ws-extraction-mode=gen-init memory-limit=0 init-part=1" %s | FileCheck --check-prefix=CHECK-INIT-PART1 %s
// RUN: vpux-opt --split-input-file --init-compiler=vpu-arch=%arch% --introduce-init-function="ws-extraction-mode=gen-main memory-limit=0" --concat-init-results="ws-extraction-mode=gen-main memory-limit=0" %s | FileCheck --check-prefix=CHECK-MAIN-PARTS %s
// RUN: vpux-opt --split-input-file --init-compiler=vpu-arch=%arch% --introduce-init-function="ws-extraction-mode=gen-all" --concat-init-results="ws-extraction-mode=gen-all" %s | FileCheck --check-prefix=CHECK-GEN-ALL %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000AABBCCDDEE",
            vpux_ow_2: "0x10000000AABBCCDDAABBCCDD"
        }
    }
#-}

// CHECK-INIT-FULL: module @TwoConstants
// CHECK-MAIN-FULL: module @TwoConstants
// CHECK-INIT-PART0: module @TwoConstants
// CHECK-INIT-PART1: module @TwoConstants
// CHECK-MAIN-PARTS: module @TwoConstants
// CHECK-GEN-ALL: module @TwoConstants
module @TwoConstants {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<4x16xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<4x16xf16>
    }

    // CHECK-INIT-FULL: net.NetworkInfo entryPoint : @init inputsInfo : {
    // CHECK-INIT-FULL-NEXT:    DataInfo "vpux_ow_1" : tensor<1x1x5x1xui8>
    // CHECK-INIT-FULL-NEXT:    DataInfo "vpux_ow_2" : tensor<2x1x1x2xf16>
    // CHECK-INIT-FULL-NEXT: } outputsInfo : {
    // CHECK-INIT-FULL-NEXT:    DataInfo "vpux_tw_0_hash_10575773572454930408_concat" : tensor<25xi8>
    // CHECK-INIT-FULL-NEXT: }

    // CHECK-MAIN-FULL: net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN-FULL-NEXT:    DataInfo "input1" : tensor<4x16xf16>
    // CHECK-MAIN-FULL-NEXT:    DataInfo "vpux_tw_0_hash_10575773572454930408_concat" : tensor<25xi8>
    // CHECK-MAIN-FULL-NEXT: } outputsInfo : {
    // CHECK-MAIN-FULL-NEXT:    DataInfo "output1" : tensor<4x16xf16>
    // CHECK-MAIN-FULL-NEXT: }


    // CHECK-INIT-PART0: net.NetworkInfo entryPoint : @init_part0 inputsInfo : {
    // CHECK-INIT-PART0-NEXT:    DataInfo "vpux_ow_1" : tensor<1x1x5x1xui8>
    // CHECK-INIT-PART0-NEXT: } outputsInfo : {
    // CHECK-INIT-PART0-NEXT:    DataInfo "vpux_tw_1_hash_16529380580407486960" : tensor<1x1x5x1xui8>
    // CHECK-INIT-PART0-NEXT: }

    // CHECK-INIT-PART1: net.NetworkInfo entryPoint : @init_part1 inputsInfo : {
    // CHECK-INIT-PART1-NEXT:    DataInfo "vpux_ow_2" : tensor<2x1x1x2xf16>
    // CHECK-INIT-PART1-NEXT: } outputsInfo : {
    // CHECK-INIT-PART1-NEXT:    DataInfo "vpux_tw_1_hash_11405229062126076964_concat" : tensor<20xi8>
    // CHECK-INIT-PART1-NEXT: }

    // CHECK-MAIN-PARTS: net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "input1" : tensor<4x16xf16>
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "vpux_tw_1_hash_16529380580407486960" : tensor<1x1x5x1xui8>
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "vpux_tw_1_hash_11405229062126076964_concat" : tensor<20xi8>
    // CHECK-MAIN-PARTS-NEXT: } outputsInfo : {
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "output1" : tensor<4x16xf16>
    // CHECK-MAIN-PARTS-NEXT: }

    func.func @main(%arg: tensor<4x16xf16>) -> tensor<4x16xf16> {
        %ov1 = const.Declare tensor<1x1x5x1xui8> = dense_resource<vpux_ow_1> : tensor<1x1x5x1xui8>, [#const.Add<1.0>]

        %ov2_0 = const.Declare tensor<2x1x1x2xf16> = dense_resource<vpux_ow_2> : tensor<2x1x1x2xf16>,
            [#const.Add<2.0>]
        %ov2_1 = const.Declare tensor<2x1x1x3xf16> = dense_resource<vpux_ow_2> : tensor<2x1x1x2xf16>,
            [#const.Add<2.0>, #const.Rescale<0.5>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 1]>]

        return %arg : tensor<4x16xf16>
    }

    // CHECK-INIT-FULL: func.func @init([[OV_1:%.+]]: tensor<1x1x5x1xui8>, [[OV_2:%.+]]: tensor<2x1x1x2xf16>)
    // CHECK-INIT-FULL-SAME: -> tensor<25xi8>
    // CHECK-INIT-FULL:     [[OV1:%.+]] = Core.ReinterpretCast({{%.+}}) {{.*}} -> tensor<5xi8>
    // CHECK-INIT-FULL:     [[OV2_0:%.+]] = Core.ReinterpretCast({{%.+}}) {{.*}} -> tensor<8xi8>
    // CHECK-INIT-FULL:     [[OV2_1:%.+]] = Core.ReinterpretCast({{%.+}}) {{.*}} -> tensor<12xi8>
    // CHECK-INIT-FULL:     [[CONCAT:%.+]] = IE.Concat([[OV1]], [[OV2_0]], [[OV2_1]]) {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-INIT-FULL:     return [[CONCAT]]

    // CHECK-MAIN-FULL: func.func @main([[IN:%.+]]: tensor<4x16xf16>, [[BLOB:%.+]]: tensor<25xi8>)
    // CHECK-MAIN-FULL-SAME: -> tensor<4x16xf16>
    // CHECK-MAIN-FULL:     [[SLICE0:%.+]] = VPU.Slice [[BLOB]] [0] [5]
    // CHECK-MAIN-FULL:     [[CAST0:%.+]] = Core.ReinterpretCast([[SLICE0]]) {{.*}} -> tensor<1x1x5x1xui8>
    // CHECK-MAIN-FULL:     [[SLICE1:%.+]] = VPU.Slice [[BLOB]] [5] [8]
    // CHECK-MAIN-FULL:     [[CAST1:%.+]] = Core.ReinterpretCast([[SLICE1]]) {{.*}} -> tensor<2x1x1x2xf16>
    // CHECK-MAIN-FULL:     [[SLICE2:%.+]] = VPU.Slice [[BLOB]] [13] [12]
    // CHECK-MAIN-FULL:     [[CAST2:%.+]] = Core.ReinterpretCast([[SLICE2]]) {{.*}} -> tensor<2x1x1x3xf16>
    // CHECK-MAIN-FULL:     return [[IN]]


    // CHECK-INIT-PART0: func.func @init_part0([[OV_1:%.+]]: tensor<1x1x5x1xui8>) -> tensor<1x1x5x1xui8>

    // CHECK-INIT-PART1: func.func @init_part1([[OV_2:%.+]]: tensor<2x1x1x2xf16>) -> tensor<20xi8>
    // CHECK-INIT-PART1:    [[OV2_0:%.+]] = Core.ReinterpretCast({{%.+}}) {{.*}} -> tensor<8xi8>
    // CHECK-INIT-PART1:    [[OV2_1:%.+]] = Core.ReinterpretCast({{%.+}}) {{.*}} -> tensor<12xi8>
    // CHECK-INIT-PART1:    [[CONCAT:%.+]] = IE.Concat([[OV2_0]], [[OV2_1]]) {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-INIT-PART1:    return [[CONCAT]]

    // CHECK-MAIN-PARTS: func.func @main([[IN:%.+]]: tensor<4x16xf16>, [[OV_1:%.+]]: tensor<1x1x5x1xui8>, [[BLOB1:%.+]]: tensor<20xi8>)
    // CHECK-MAIN-PARTS-SAME: -> tensor<4x16xf16>
    // CHECK-MAIN-PARTS:    [[SLICE10:%.+]] = VPU.Slice [[BLOB1]] [0] [8]
    // CHECK-MAIN-PARTS:    [[CAST10:%.+]] = Core.ReinterpretCast([[SLICE10]]) {{.*}} -> tensor<2x1x1x2xf16>
    // CHECK-MAIN-PARTS:    [[SLICE11:%.+]] = VPU.Slice [[BLOB1]] [8] [12]
    // CHECK-MAIN-PARTS:    [[CAST11:%.+]] = Core.ReinterpretCast([[SLICE11]]) {{.*}} -> tensor<2x1x1x3xf16>
    // CHECK-MAIN-PARTS:    return [[IN]]

    // CHECK-GEN-ALL:   func.func @wrapper_main([[IN:%.+]]: tensor<4x16xf16>) -> tensor<4x16xf16>
    // CHECK-GEN-ALL:       [[OV1:%.+]] = const.Declare tensor<1x1x5x1xui8> = dense_resource<vpux_ow_1>
    // CHECK-GEN-ALL:       [[OV2:%.+]] = const.Declare tensor<2x1x1x2xf16> = dense_resource<vpux_ow_2>
    // CHECK-GEN-ALL:       [[CALL_INIT:%.+]] = call @init([[OV1]], [[OV2]])
    // CHECK-GEN-ALL-SAME:      -> tensor<25xi8>
    // CHECK-GEN-ALL:       [[CALL_MAIN:%.+]] = call @main([[IN]], [[CALL_INIT]])
    // CHECK-GEN-ALL:       return [[CALL_MAIN]]
}

// -----

// Note: this tests both quantized types *and* that generated IR correctly
//       interacts with (pre-existing) IR in main

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000AABBCCDD",

            // Note: required to successfully compile "init-part=1"
            vpux_ow_dummy: "0x10000000AABBCCDD"
        }
    }
#-}

!qElemType = !quant.uniform<u8:f16, 0.5:120>
// CHECK-INIT-FULL: [[QTYPE:!.+]] = !quant.uniform<u8:f16, 5.000000e-01:120>
// CHECK-MAIN-FULL: [[QTYPE:!.+]] = !quant.uniform<u8:f16, 5.000000e-01:120>
// CHECK-MAIN-PARTS: [[QTYPE:!.+]] = !quant.uniform<u8:f16, 5.000000e-01:120>

// CHECK-INIT-FULL: module @QuantizedType
// CHECK-MAIN-FULL: module @QuantizedType
// CHECK-INIT-PART0: module @QuantizedType
// CHECK-INIT-PART1: module @QuantizedType
// CHECK-MAIN-PARTS: module @QuantizedType
// CHECK-GEN-ALL: module @QuantizedType
module @QuantizedType {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<4x16xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<4x16xf16>
    }

    // CHECK-INIT-FULL: net.NetworkInfo entryPoint : @init inputsInfo : {
    // CHECK-INIT-FULL-NEXT:    DataInfo "vpux_ow_dummy" : tensor<2xf16>
    // CHECK-INIT-FULL-NEXT:    DataInfo "vpux_ow_1" : tensor<2xf16>
    // CHECK-INIT-FULL-NEXT: } outputsInfo : {
    // CHECK-INIT-FULL-NEXT:    DataInfo "vpux_tw_0_hash_2864067019402973834_concat" : tensor<11xi8>
    // CHECK-INIT-FULL-NEXT: }

    // CHECK-MAIN-FULL: net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN-FULL-NEXT:    DataInfo "input1" : tensor<4x16xf16>
    // CHECK-MAIN-FULL-NEXT:    DataInfo "vpux_tw_0_hash_2864067019402973834_concat" : tensor<11xi8>
    // CHECK-MAIN-FULL-NEXT: } outputsInfo : {
    // CHECK-MAIN-FULL-NEXT:    DataInfo "output1" : tensor<4x16xf16>
    // CHECK-MAIN-FULL-NEXT: }

    // CHECK-INIT-PART0: net.NetworkInfo entryPoint : @init_part0 inputsInfo : {
    // CHECK-INIT-PART0-NEXT:    DataInfo "vpux_ow_dummy" : tensor<2xf16>
    // CHECK-INIT-PART0-NEXT: } outputsInfo : {
    // CHECK-INIT-PART0-NEXT:    DataInfo "vpux_tw_dummy_hash_16529380580407486960" : tensor<2xf16>
    // CHECK-INIT-PART0-NEXT: }

    // CHECK-INIT-PART1: net.NetworkInfo entryPoint : @init_part1 inputsInfo : {
    // CHECK-INIT-PART1-NEXT:    DataInfo "vpux_ow_1" : tensor<2xf16>
    // CHECK-INIT-PART1-NEXT: } outputsInfo : {
    // CHECK-INIT-PART1-NEXT:    DataInfo "vpux_tw_1_hash_8290054905247884848_concat" : tensor<7xi8>
    // CHECK-INIT-PART1-NEXT: }

    // CHECK-MAIN-PARTS: net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "input1" : tensor<4x16xf16>
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "vpux_tw_dummy_hash_16529380580407486960" : tensor<2xf16>
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "vpux_tw_1_hash_8290054905247884848_concat" : tensor<7xi8>
    // CHECK-MAIN-PARTS-NEXT: } outputsInfo : {
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "output1" : tensor<4x16xf16>
    // CHECK-MAIN-PARTS-NEXT: }

    func.func @main(%arg: tensor<4x16xf16>) -> tensor<4x16xf16> {
        %ov1_0 = const.Declare tensor<5x!qElemType> = dense_resource<vpux_ow_1> : tensor<2xf16>,
            [#const.Add<1.0>, #const.CastElemType<!qElemType>, #const.PadWithZero<[0], [3]>]
        %ov1_1 = const.Declare tensor<2x!qElemType> = dense_resource<vpux_ow_1> : tensor<2xf16>,
            [#const.Add<1.0>, #const.CastElemType<!qElemType>]

        %dummy = const.Declare tensor<2xf16> = dense_resource<vpux_ow_dummy> : tensor<2xf16>, [#const.Add<1.0>]

        return %arg : tensor<4x16xf16>
    }

    // CHECK-INIT-FULL: func.func @init([[OV_DUMMY:%.+]]: tensor<2xf16>, [[OV_1:%.+]]: tensor<2xf16>) -> tensor<11xi8>
    // CHECK-INIT-FULL:     [[BOUNDARY_CAST0:%.+]] = IE.QuantizeCast({{%.+}}) {dstElemType = ui8}
    // CHECK-INIT-FULL:     [[BOUNDARY_CAST1:%.+]] = IE.QuantizeCast({{%.+}}) {dstElemType = ui8}
    // CHECK-INIT-FULL:     [[OV1_1:%.+]] = Core.ReinterpretCast([[BOUNDARY_CAST0]]) {{.*}} -> tensor<2xi8>
    // CHECK-INIT-FULL:     [[OV1_0:%.+]] = Core.ReinterpretCast([[BOUNDARY_CAST1]]) {{.*}} -> tensor<5xi8>
    // CHECK-INIT-FULL:     [[CONCAT:%.+]] = IE.Concat({{%.+}}, [[OV1_1]], [[OV1_0]]) {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-INIT-FULL:     return [[CONCAT]]

    // CHECK-MAIN-FULL: func.func @main([[IN:%.+]]: tensor<4x16xf16>, [[BLOB:%.+]]: tensor<11xi8>)
    // CHECK-MAIN-FULL-SAME: -> tensor<4x16xf16>
    // CHECK-MAIN-FULL:     [[SLICE_OV1_1:%.+]] = VPU.Slice [[BLOB]] [4] [2]
    // CHECK-MAIN-FULL:     [[CAST_OV1_1:%.+]] = Core.ReinterpretCast([[SLICE_OV1_1]]) {{.*}} -> tensor<2xui8>
    // CHECK-MAIN-FULL:     [[SLICE_OV1_0:%.+]] = VPU.Slice [[BLOB]] [6] [5]
    // CHECK-MAIN-FULL:     [[CAST_OV1_0:%.+]] = Core.ReinterpretCast([[SLICE_OV1_0]]) {{.*}} -> tensor<5xui8>
    // CHECK-MAIN-FULL:     [[BOUNDARY_CAST0:%.+]] = VPU.QuantizeCast([[CAST_OV1_1]]) {dstElemType = [[QTYPE]]}
    // CHECK-MAIN-FULL:     [[BOUNDARY_CAST1:%.+]] = VPU.QuantizeCast([[CAST_OV1_0]]) {dstElemType = [[QTYPE]]}
    // CHECK-MAIN-FULL:     return [[IN]]

    // CHECK-INIT-PART0: func.func @init_part0([[DUMMY:%.+]]: tensor<2xf16>) -> tensor<2xf16>

    // CHECK-INIT-PART1: func.func @init_part1([[OV_1:%.+]]: tensor<2xf16>) -> tensor<7xi8>
    // CHECK-INIT-PART1:    [[OV1_1:%.+]] = Core.ReinterpretCast({{.*}}) {{.*}} -> tensor<2xi8>
    // CHECK-INIT-PART1:    [[OV1_0:%.+]] = Core.ReinterpretCast({{.*}}) {{.*}} -> tensor<5xi8>
    // CHECK-INIT-PART1:    [[CONCAT:%.+]] = IE.Concat([[OV1_1]], [[OV1_0]]) {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-INIT-PART1:    return [[CONCAT]]

    // CHECK-MAIN-PARTS: func.func @main([[IN:%.+]]: tensor<4x16xf16>, [[DUMMY:%.+]]: tensor<2xf16>, [[BLOB:%.+]]: tensor<7xi8>)
    // CHECK-MAIN-PARTS-SAME: -> tensor<4x16xf16>
    // CHECK-MAIN-PARTS:    [[SLICE_OV1_1:%.+]] = VPU.Slice [[BLOB]] [0] [2]
    // CHECK-MAIN-PARTS:    [[CAST_OV1_1:%.+]] = Core.ReinterpretCast([[SLICE_OV1_1]]) {{.*}} -> tensor<2xui8>
    // CHECK-MAIN-PARTS:    [[SLICE_OV1_0:%.+]] = VPU.Slice [[BLOB]] [2] [5]
    // CHECK-MAIN-PARTS:    [[CAST_OV1_0:%.+]] = Core.ReinterpretCast([[SLICE_OV1_0]]) {{.*}} -> tensor<5xui8>
    // CHECK-MAIN-PARTS:    [[BOUNDARY_CAST0:%.+]] = VPU.QuantizeCast([[CAST_OV1_1]]) {dstElemType = [[QTYPE]]}
    // CHECK-MAIN-PARTS:    [[BOUNDARY_CAST1:%.+]] = VPU.QuantizeCast([[CAST_OV1_0]]) {dstElemType = [[QTYPE]]}
    // CHECK-MAIN-PARTS:    return [[IN]]

    // CHECK-GEN-ALL:   func.func @wrapper_main([[IN:%.+]]: tensor<4x16xf16>) -> tensor<4x16xf16>
    // CHECK-GEN-ALL:       [[OV1:%.+]] = const.Declare tensor<2xf16> = dense_resource<vpux_ow_1>
    // CHECK-GEN-ALL:       [[OVDUMMY:%.+]] = const.Declare tensor<2xf16> = dense_resource<vpux_ow_dummy>
    // CHECK-GEN-ALL:       [[CALL_INIT:%.+]] = call @init([[OV1]], [[OVDUMMY]])
    // CHECK-GEN-ALL-SAME:      -> tensor<11xi8>
    // CHECK-GEN-ALL:       [[CALL_MAIN:%.+]] = call @main([[IN]], [[CALL_INIT]])
    // CHECK-GEN-ALL:       return [[CALL_MAIN]]
}


// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000AABBCCDD",

            // Note: required to successfully compile "init-part=1"
            vpux_ow_dummy: "0x10000000AABBCCDD"
        }
    }
#-}

// CHECK-INIT-FULL: module @SimpleOutlining
// CHECK-MAIN-FULL: module @SimpleOutlining
// CHECK-INIT-PART0: module @SimpleOutlining
// CHECK-INIT-PART1: module @SimpleOutlining
// CHECK-MAIN-PARTS: module @SimpleOutlining
// CHECK-GEN-ALL: module @SimpleOutlining
module @SimpleOutlining {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<4x16xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<4x16xf16>
    }

    // CHECK-INIT-FULL: net.NetworkInfo entryPoint : @init inputsInfo : {
    // CHECK-INIT-FULL-NEXT:    DataInfo "vpux_ow_dummy" : tensor<1x1x1x2xf16>
    // CHECK-INIT-FULL-NEXT:    DataInfo "vpux_ow_1" : tensor<1x1x1x2xf16>
    // CHECK-INIT-FULL-NEXT: } outputsInfo : {
    // CHECK-INIT-FULL-NEXT:    DataInfo "vpux_tw_0_hash_14095452562947179690_concat" : tensor<24xi8>
    // CHECK-INIT-FULL-NEXT: }

    // CHECK-MAIN-FULL: net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN-FULL-NEXT:    DataInfo "input1" : tensor<4x16xf16>
    // CHECK-MAIN-FULL-NEXT:    DataInfo "vpux_tw_0_hash_14095452562947179690_concat" : tensor<24xi8>
    // CHECK-MAIN-FULL-NEXT: } outputsInfo : {
    // CHECK-MAIN-FULL-NEXT:    DataInfo "output1" : tensor<4x16xf16>
    // CHECK-MAIN-FULL-NEXT: }

    func.func private @main_part1() -> tensor<1x1x1x3xf16> {
        %ov_internal = const.Declare tensor<1x1x1x3xf16> = dense_resource<vpux_ow_1> : tensor<1x1x1x2xf16>,
            [#const.Add<42.0>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 1]>]
        return %ov_internal : tensor<1x1x1x3xf16>
    }

    func.func @main(%arg: tensor<4x16xf16>) -> tensor<4x16xf16> {
        %ov1_0 = const.Declare tensor<1x1x1x5xf16> = dense_resource<vpux_ow_1> : tensor<1x1x1x2xf16>,
            [#const.Add<1.0>, #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 3]>]
        %ov1_1 = const.Declare tensor<1x1x1x2xf16> = dense_resource<vpux_ow_1> : tensor<1x1x1x2xf16>,
            [#const.Add<1.0>]
        %ov1_2 = func.call @main_part1() : () -> tensor<1x1x1x3xf16>

        %dummy = const.Declare tensor<1x1x1x2xf16> = dense_resource<vpux_ow_dummy> : tensor<1x1x1x2xf16>, [#const.Add<1.0>]

        return %arg : tensor<4x16xf16>
    }

    // CHECK-INIT-FULL: func.func @init([[OV_DUMMY:%.+]]: tensor<1x1x1x2xf16>, [[OV_1:%.+]]: tensor<1x1x1x2xf16>)
    // CHECK-INIT-FULL-SAME: -> tensor<24xi8>
    // CHECK-INIT-FULL:     [[OV1_1:%.+]] = Core.ReinterpretCast({{%.+}}) {{.*}} -> tensor<4xi8>
    // CHECK-INIT-FULL:     [[OV1_0:%.+]] = Core.ReinterpretCast({{%.+}}) {{.*}} -> tensor<10xi8>
    // CHECK-INIT-FULL:     [[OV1_2:%.+]] = Core.ReinterpretCast({{%.+}}) {{.*}} -> tensor<6xi8>
    // CHECK-INIT-FULL:     [[CONCAT:%.+]] = IE.Concat([[OV1_1]], [[OV1_0]], {{%.+}}, [[OV1_2]]) {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-INIT-FULL:     return [[CONCAT]]

    // CHECK-MAIN-FULL: func.func private @main_part1([[OV1_2:%.+]]: tensor<1x1x1x3xf16>) -> tensor<1x1x1x3xf16>
    // CHECK-MAIN-FULL:     return [[OV1_2]]

    // CHECK-MAIN-FULL: func.func @main([[IN:%.+]]: tensor<4x16xf16>, [[BLOB:%.+]]: tensor<24xi8>)
    // CHECK-MAIN-FULL-SAME: -> tensor<4x16xf16>
    // CHECK-MAIN-FULL:     [[SLICE_OV1_1:%.+]] = VPU.Slice [[BLOB]] [0] [4]
    // CHECK-MAIN-FULL:     [[CAST_OV1_1:%.+]] = Core.ReinterpretCast([[SLICE_OV1_1]]) {{.*}} -> tensor<1x1x1x2xf16>
    // CHECK-MAIN-FULL:     [[SLICE_OV1_0:%.+]] = VPU.Slice [[BLOB]] [4] [10]
    // CHECK-MAIN-FULL:     [[CAST_OV1_0:%.+]] = Core.ReinterpretCast([[SLICE_OV1_0]]) {{.*}} -> tensor<1x1x1x5xf16>
    // CHECK-MAIN-FULL:     [[SLICE_DUMMY:%.+]] = VPU.Slice [[BLOB]] [14] [4]
    // CHECK-MAIN-FULL:     [[CAST_DUMMY:%.+]] = Core.ReinterpretCast([[SLICE_DUMMY]]) {{.*}} -> tensor<1x1x1x2xf16>
    // CHECK-MAIN-FULL:     [[SLICE_OV1_2:%.+]] = VPU.Slice [[BLOB]] [18] [6]
    // CHECK-MAIN-FULL:     [[CAST_OV1_2:%.+]] = Core.ReinterpretCast([[SLICE_OV1_2]]) {{.*}} -> tensor<1x1x1x3xf16>
    // CHECK-MAIN-FULL:     {{%.+}} = call @main_part1([[CAST_OV1_2]])
    // CHECK-MAIN-FULL:     return [[IN]]

    // CHECK-GEN-ALL:   func.func @wrapper_main([[IN:%.+]]: tensor<4x16xf16>) -> tensor<4x16xf16>
    // CHECK-GEN-ALL:       [[OV1:%.+]] = const.Declare tensor<1x1x1x2xf16> = dense_resource<vpux_ow_1>
    // CHECK-GEN-ALL:       [[OVDUMMY:%.+]] = const.Declare tensor<1x1x1x2xf16> = dense_resource<vpux_ow_dummy>
    // CHECK-GEN-ALL:       [[CALL_INIT:%.+]] = call @init([[OV1]], [[OVDUMMY]])
    // CHECK-GEN-ALL-SAME:      -> tensor<24xi8>
    // CHECK-GEN-ALL:       [[CALL_MAIN:%.+]] = call @main([[IN]], [[CALL_INIT]])
    // CHECK-GEN-ALL:       return [[CALL_MAIN]]
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

// CHECK-INIT-FULL: module @SingleConstantInTheBeginning
// CHECK-MAIN-FULL: module @SingleConstantInTheBeginning
// CHECK-INIT-PART0: module @SingleConstantInTheBeginning
// CHECK-INIT-PART1: module @SingleConstantInTheBeginning
// CHECK-MAIN-PARTS: module @SingleConstantInTheBeginning
// CHECK-GEN-ALL: module @SingleConstantInTheBeginning
module @SingleConstantInTheBeginning {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<4x16xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<4x16xf16>
    }

    // CHECK-INIT-PART0: net.NetworkInfo entryPoint : @init_part0 inputsInfo : {
    // CHECK-INIT-PART0-NEXT:    DataInfo "vpux_ow_2" : tensor<2xf16>
    // CHECK-INIT-PART0-NEXT: } outputsInfo : {
    // CHECK-INIT-PART0-NEXT:    DataInfo "vpux_tw_2_hash_8938469330746701159" : tensor<12xf16>
    // CHECK-INIT-PART0-NEXT: }

    // CHECK-INIT-PART0: func.func @init_part0([[OV_2:%.+]]: tensor<2xf16>) -> tensor<12xf16>
    // CHECK-INIT-PART0:    [[OUT:%.+]] = IE.Pad([[OV_2]])
    // CHECK-INIT-PART0:    return [[OUT]]

    // CHECK-INIT-PART1: net.NetworkInfo entryPoint : @init_part1 inputsInfo : {
    // CHECK-INIT-PART1-NEXT:    DataInfo "vpux_ow_1" : tensor<2xf16>
    // CHECK-INIT-PART1-NEXT: } outputsInfo : {
    // CHECK-INIT-PART1-NEXT:    DataInfo "vpux_tw_1_hash_8692743050400081167_concat" : tensor<208xi8>
    // CHECK-INIT-PART1-NEXT: }

    // CHECK-INIT-PART1: func.func @init_part1([[OV_1:%.+]]: tensor<2xf16>) -> tensor<208xi8>

    // CHECK-MAIN-PARTS: net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "input1" : tensor<4x16xf16>
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "vpux_tw_2_hash_8938469330746701159" : tensor<12xf16>
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "vpux_tw_1_hash_8692743050400081167_concat" : tensor<208xi8>
    // CHECK-MAIN-PARTS-NEXT: } outputsInfo : {
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "output1" : tensor<4x16xf16>
    // CHECK-MAIN-PARTS-NEXT: }

    // CHECK-MAIN-PARTS: func.func @main([[IN:%.+]]: tensor<4x16xf16>, [[OV_2:%.+]]: tensor<12xf16>, [[BLOB0:%.+]]: tensor<208xi8>)

    func.func @main(%arg: tensor<4x16xf16>) -> tensor<4x16xf16> {
        %ov2_single = const.Declare tensor<12xf16> = dense_resource<vpux_ow_2> : tensor<2xf16>,
            [#const.PadWithZero<[0], [10]>]

        %ov1_0 = const.Declare tensor<102xf16> = dense_resource<vpux_ow_1> : tensor<2xf16>,
            [#const.Add<1.0>, #const.PadWithZero<[0], [100]>]
        %ov1_1 = const.Declare tensor<2xf16> = dense_resource<vpux_ow_1> : tensor<2xf16>, [#const.Add<2.0>]

        return %arg : tensor<4x16xf16>
    }
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000AABBCCDD",
            vpux_ow_2: "0x10000000AABBCCDD",
            vpux_ow_3: "0x10000000AABBCCDD"
        }
    }
#-}

// CHECK-INIT-FULL: module @SingleConstantInTheMiddle
// CHECK-MAIN-FULL: module @SingleConstantInTheMiddle
// CHECK-INIT-PART0: module @SingleConstantInTheMiddle
// CHECK-INIT-PART1: module @SingleConstantInTheMiddle
// CHECK-MAIN-PARTS: module @SingleConstantInTheMiddle
// CHECK-GEN-ALL: module @SingleConstantInTheMiddle
module @SingleConstantInTheMiddle {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<4x16xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<4x16xf16>
    }

    // CHECK-INIT-PART0: net.NetworkInfo entryPoint : @init_part0 inputsInfo : {
    // CHECK-INIT-PART0-NEXT:    DataInfo "vpux_ow_1" : tensor<2xf16>
    // CHECK-INIT-PART0-NEXT: } outputsInfo : {
    // CHECK-INIT-PART0-NEXT:    DataInfo "vpux_tw_0_hash_13109616749475806820_concat" : tensor<8xi8>
    // CHECK-INIT-PART0-NEXT: }

    // CHECK-INIT-PART0: func.func @init_part0([[OV_1:%.+]]: tensor<2xf16>) -> tensor<8xi8>

    // CHECK-INIT-PART1: net.NetworkInfo entryPoint : @init_part1 inputsInfo : {
    // CHECK-INIT-PART1-NEXT:    DataInfo "vpux_ow_2" : tensor<2xf16>
    // CHECK-INIT-PART1-NEXT: } outputsInfo : {
    // CHECK-INIT-PART1-NEXT:    DataInfo "vpux_tw_2_hash_8938469330746701159" : tensor<12xf16>
    // CHECK-INIT-PART1-NEXT: }

    // CHECK-INIT-PART1: func.func @init_part1([[OV_2:%.+]]: tensor<2xf16>) -> tensor<12xf16>
    // CHECK-INIT-PART1:    [[OUT:%.+]] = IE.Pad([[OV_2]])
    // CHECK-INIT-PART1:    return [[OUT]]

    // CHECK-MAIN-PARTS: net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "input1" : tensor<4x16xf16>
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "vpux_tw_0_hash_13109616749475806820_concat" : tensor<8xi8>
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "vpux_tw_2_hash_8938469330746701159" : tensor<12xf16>
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "vpux_tw_2_hash_2332981286748766850_concat" : tensor<108xi8>
    // CHECK-MAIN-PARTS-NEXT: } outputsInfo : {
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "output1" : tensor<4x16xf16>
    // CHECK-MAIN-PARTS-NEXT: }

    // CHECK-MAIN-PARTS: func.func @main([[IN:%.+]]: tensor<4x16xf16>, [[BLOB0:%.+]]: tensor<8xi8>, [[OV_2:%.+]]: tensor<12xf16>, [[BLOB2:%.+]]: tensor<108xi8>)

    func.func @main(%arg: tensor<4x16xf16>) -> tensor<4x16xf16> {
        %ov1_0 = const.Declare tensor<2xf16> = dense_resource<vpux_ow_1> : tensor<2xf16>, [#const.Add<1.0>]
        %ov1_1 = const.Declare tensor<2xf16> = dense_resource<vpux_ow_1> : tensor<2xf16>, [#const.Add<2.0>]

        %ov2_single = const.Declare tensor<12xf16> = dense_resource<vpux_ow_2> : tensor<2xf16>,
            [#const.PadWithZero<[0], [10]>]

        %ov3_0 = const.Declare tensor<2xf16> = dense_resource<vpux_ow_3> : tensor<2xf16>, [#const.Add<1.0>]
        %ov3_1 = const.Declare tensor<52xf16> = dense_resource<vpux_ow_3> : tensor<2xf16>,
            [#const.PadWithZero<[0], [50]>]

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

// CHECK-INIT-FULL: module @SingleConstantInTheEnd
// CHECK-MAIN-FULL: module @SingleConstantInTheEnd
// CHECK-INIT-PART0: module @SingleConstantInTheEnd
// CHECK-INIT-PART1: module @SingleConstantInTheEnd
// CHECK-MAIN-PARTS: module @SingleConstantInTheEnd
// CHECK-GEN-ALL: module @SingleConstantInTheEnd
module @SingleConstantInTheEnd {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<4x16xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<4x16xf16>
    }

    // CHECK-INIT-PART0: net.NetworkInfo entryPoint : @init_part0 inputsInfo : {
    // CHECK-INIT-PART0-NEXT:    DataInfo "vpux_ow_1" : tensor<2xf16>
    // CHECK-INIT-PART0-NEXT: } outputsInfo : {
    // CHECK-INIT-PART0-NEXT:    DataInfo "vpux_tw_0_hash_13109616749475806820_concat" : tensor<8xi8>
    // CHECK-INIT-PART0-NEXT: }

    // CHECK-INIT-PART0: func.func @init_part0([[OV_1:%.+]]: tensor<2xf16>) -> tensor<8xi8>

    // CHECK-INIT-PART1: net.NetworkInfo entryPoint : @init_part1 inputsInfo : {
    // CHECK-INIT-PART1-NEXT:    DataInfo "vpux_ow_2" : tensor<2xf16>
    // CHECK-INIT-PART1-NEXT: } outputsInfo : {
    // CHECK-INIT-PART1-NEXT:    DataInfo "vpux_tw_2_hash_8938469330746701159" : tensor<12xf16>
    // CHECK-INIT-PART1-NEXT: }

    // CHECK-INIT-PART1: func.func @init_part1([[OV_2:%.+]]: tensor<2xf16>) -> tensor<12xf16>
    // CHECK-INIT-PART1:    [[OUT:%.+]] = IE.Pad([[OV_2]])
    // CHECK-INIT-PART1:    return [[OUT]]

    // CHECK-MAIN-PARTS: net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "input1" : tensor<4x16xf16>
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "vpux_tw_0_hash_13109616749475806820_concat" : tensor<8xi8>
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "vpux_tw_2_hash_8938469330746701159" : tensor<12xf16>
    // CHECK-MAIN-PARTS-NEXT: } outputsInfo : {
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "output1" : tensor<4x16xf16>
    // CHECK-MAIN-PARTS-NEXT: }

    // CHECK-MAIN-PARTS: func.func @main([[IN:%.+]]: tensor<4x16xf16>, [[BLOB0:%.+]]: tensor<8xi8>, [[OV_2:%.+]]: tensor<12xf16>)

    func.func @main(%arg: tensor<4x16xf16>) -> tensor<4x16xf16> {
        %ov1_0 = const.Declare tensor<2xf16> = dense_resource<vpux_ow_1> : tensor<2xf16>, [#const.Add<1.0>]
        %ov1_1 = const.Declare tensor<2xf16> = dense_resource<vpux_ow_1> : tensor<2xf16>, [#const.Add<2.0>]

        %ov2_single = const.Declare tensor<12xf16> = dense_resource<vpux_ow_2> : tensor<2xf16>,
            [#const.PadWithZero<[0], [10]>]

        return %arg : tensor<4x16xf16>
    }
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000AABBCCDD",
            vpux_ow_2: "0x10000000AABBCCDD",
            vpux_ow_3: "0x10000000AABBCCDD"
        }
    }
#-}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-INIT-FULL: module @SingleConstantWithLayout
// CHECK-MAIN-FULL: module @SingleConstantWithLayout
// CHECK-INIT-PART0: module @SingleConstantWithLayout
// CHECK-INIT-PART1: module @SingleConstantWithLayout
// CHECK-MAIN-PARTS: module @SingleConstantWithLayout
// CHECK-GEN-ALL: module @SingleConstantWithLayout
module @SingleConstantWithLayout {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<4x16xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<4x16xf16>
    }

    // CHECK-INIT-FULL: net.NetworkInfo entryPoint : @init inputsInfo : {
    // CHECK-INIT-FULL-NEXT:    DataInfo "vpux_ow_1" : tensor<2xf16>
    // CHECK-INIT-FULL-NEXT:    DataInfo "vpux_ow_2" : tensor<1x1x1x2xf16>
    // CHECK-INIT-FULL-NEXT:    DataInfo "vpux_ow_3" : tensor<2xf16>
    // CHECK-INIT-FULL-NEXT: } outputsInfo : {
    // CHECK-INIT-FULL-NEXT:    DataInfo "vpux_tw_0_hash_12906375107675847465_concat" : tensor<140xi8>
    // CHECK-INIT-FULL-NEXT: }

    // CHECK-INIT-FULL: func.func @init({{%.+}}: tensor<2xf16>, {{%.+}}: tensor<1x1x1x2xf16>, {{%.+}}: tensor<2xf16>)
    // CHECK-INIT-FULL-SAME: -> tensor<140xi8>

    // CHECK-MAIN-FULL: net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN-FULL-NEXT:    DataInfo "input1" : tensor<4x16xf16>
    // CHECK-MAIN-FULL-NEXT:    DataInfo "vpux_tw_0_hash_12906375107675847465_concat" : tensor<140xi8>
    // CHECK-MAIN-FULL-NEXT: } outputsInfo : {
    // CHECK-MAIN-FULL-NEXT:    DataInfo "output1" : tensor<4x16xf16>
    // CHECK-MAIN-FULL-NEXT: }

    // CHECK-MAIN-FULL: func.func @main([[IN:%.+]]: tensor<4x16xf16>, [[BLOB:%.+]]: tensor<140xi8>)


    // CHECK-INIT-PART0: net.NetworkInfo entryPoint : @init_part0 inputsInfo : {
    // CHECK-INIT-PART0-NEXT:    DataInfo "vpux_ow_1" : tensor<2xf16>
    // CHECK-INIT-PART0-NEXT: } outputsInfo : {
    // CHECK-INIT-PART0-NEXT:    DataInfo "vpux_tw_0_hash_13109616749475806820_concat" : tensor<8xi8>
    // CHECK-INIT-PART0-NEXT: }

    // CHECK-INIT-PART0: func.func @init_part0([[OV_1:%.+]]: tensor<2xf16>) -> tensor<8xi8>

    // CHECK-INIT-PART1: net.NetworkInfo entryPoint : @init_part1 inputsInfo : {
    // CHECK-INIT-PART1-NEXT:    DataInfo "vpux_ow_2" : tensor<1x1x1x2xf16>
    // CHECK-INIT-PART1-NEXT: } outputsInfo : {
    // CHECK-INIT-PART1-NEXT:    DataInfo "vpux_tw_2_hash_10502553507830482865" : tensor<1x1x1x12xf16>
    // CHECK-INIT-PART1-NEXT: }

    // CHECK-INIT-PART1: func.func @init_part1([[OV_2:%.+]]: tensor<1x1x1x2xf16>)
    // CHECK-INIT-PART1-SAME:   -> tensor<1x1x1x12xf16>
    // CHECK-INIT-PART1:    [[OUT:%.+]] = IE.Pad([[OV_2]])
    // CHECK-INIT-PART1:    return [[OUT]]

    // CHECK-MAIN-PARTS: net.NetworkInfo entryPoint : @main inputsInfo : {
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "input1" : tensor<4x16xf16>
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "vpux_tw_0_hash_13109616749475806820_concat" : tensor<8xi8>
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "vpux_tw_2_hash_10502553507830482865" : tensor<1x1x1x12xf16>
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "vpux_tw_2_hash_2332981286748766850_concat" : tensor<108xi8>
    // CHECK-MAIN-PARTS-NEXT: } outputsInfo : {
    // CHECK-MAIN-PARTS-NEXT:    DataInfo "output1" : tensor<4x16xf16>
    // CHECK-MAIN-PARTS-NEXT: }

    // CHECK-MAIN-PARTS: func.func @main([[IN:%.+]]: tensor<4x16xf16>, [[BLOB0:%.+]]: tensor<8xi8>, [[OV_2:%.+]]: tensor<1x1x1x12xf16>, [[BLOB2:%.+]]: tensor<108xi8>)

    func.func @main(%arg: tensor<4x16xf16>) -> tensor<4x16xf16> {
        %ov1_0 = const.Declare tensor<2xf16> = dense_resource<vpux_ow_1> : tensor<2xf16>, [#const.Add<1.0>]
        %ov1_1 = const.Declare tensor<2xf16> = dense_resource<vpux_ow_1> : tensor<2xf16>, [#const.Add<2.0>]

        %ov2_single_reordered = const.Declare tensor<1x1x1x12xf16, {order = #NHWC}> = dense_resource<vpux_ow_2>
            : tensor<1x1x1x2xf16>, [#const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 10]>, #const.Reorder<#NHWC>]

        %ov3_0 = const.Declare tensor<2xf16> = dense_resource<vpux_ow_3> : tensor<2xf16>, [#const.Add<1.0>]
        %ov3_1 = const.Declare tensor<52xf16> = dense_resource<vpux_ow_3> : tensor<2xf16>,
            [#const.PadWithZero<[0], [50]>]

        return %arg : tensor<4x16xf16>
    }
}

// -----

{-#
    dialect_resources: {
        builtin: {
            vpux_ow_1: "0x10000000112233445566778899AA",
            vpux_ow_2: "0x10000000112233445566",
            vpux_ow_3: "0x1000000011223344"
        }
    }
#-}

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType1 = !quant.uniform<u8:f16:0, {4.6881728020368838E-4:128,8.9274726661981319E-5:128,7.3915015833050598E-4:128,0.004798228366702211:128,0.0011995570916755527:128,8.4776390416949401E-4:128,0.0020088736917458329:128,0.0024721641166537416:128,5.6757888957565902E-4:128,0.0025548259417215984:128}>

!qElemType2 = !quant.uniform<i8:f16:0, {4.6881728020368838E-4,8.9274726661981319E-5,7.3915015833050598E-4,0.004798228366702211,0.0011995570916755527,8.4776390416949401E-4,0.0020088736917458329,0.0024721641166537416,5.6757888957565902E-4,0.0025548259417215984}>

!qElemType3 = !quant.uniform<u8:f16:0, {3.3872110732630188E-5:128,7.9393763752544626E-4:128,4.8323503019762976E-4:128,0.001715712687548469:128,0.0012831800708583757:128,0.0010909433458365645:128}>

!qElemType4 = !quant.uniform<i8:f16:0, {3.3872110732630188E-5,7.9393763752544626E-4,4.8323503019762976E-4,0.001715712687548469,0.0012831800708583757,0.0010909433458365645}>

!qElemType5 = !quant.uniform<u8:f16:0, {0.0011582261791416243:128,0.0016320897083656461:128,0.0020050289584141153:128,0.0031276913250193874:128}>

!qElemType6 = !quant.uniform<i8:f16:0, {0.0011582261791416243,0.0016320897083656461,0.0020050289584141153,0.0031276913250193874}>

// CHECK-INIT-FULL: module @GenAllSpecialCase
// CHECK-MAIN-FULL: module @GenAllSpecialCase
// CHECK-INIT-PART0: module @GenAllSpecialCase
// CHECK-INIT-PART1: module @GenAllSpecialCase
// CHECK-MAIN-PARTS: module @GenAllSpecialCase
// CHECK-GEN-ALL: module @GenAllSpecialCase
module @GenAllSpecialCase {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input1" : tensor<4x16xf16>
    } outputsInfo : {
        DataInfo "output1" : tensor<4x16xf16>
        DataInfo "cst1" : tensor<10x1x1x1xsi8>
        DataInfo "cst2" : tensor<6x1x1x1xsi8>
        DataInfo "cst3" : tensor<4x1x1x1xsi8>
    }

    func.func @main(%arg: tensor<4x16xf16>)
            -> (tensor<4x16xf16>, tensor<10x1x1x1xsi8, {order = #NHWC}>,
                tensor<6x1x1x1xsi8, {order = #NHWC}>, tensor<4x1x1x1xsi8, {order = #NHWC}>) {
        %cst1 = const.Declare tensor<10x1x1x1x!qElemType1, {order = #NHWC}> = dense_resource<vpux_ow_1>
            : tensor<10x1x1x1xsi8>,
            [#const.CastElemType<f16>, #const.CastElemType<!qElemType2>,
             #const.ConvertElemType<!qElemType1>, #const.Reorder<#NHWC>]
        %cst2 = const.Declare tensor<6x1x1x1x!qElemType3, {order = #NHWC}> = dense_resource<vpux_ow_2>
            : tensor<6x1x1x1xsi8>,
            [#const.CastElemType<f16>, #const.CastElemType<!qElemType4>,
             #const.ConvertElemType<!qElemType3>, #const.Reorder<#NHWC>]
        %cst3 = const.Declare tensor<4x1x1x1x!qElemType5, {order = #NHWC}> = dense_resource<vpux_ow_3>
            : tensor<4x1x1x1xsi8>,
            [#const.CastElemType<f16>, #const.CastElemType<!qElemType6>,
             #const.ConvertElemType<!qElemType5>, #const.Reorder<#NHWC>]

        %cast1 = VPU.QuantizeCast(%cst1) {dstElemType = si8}
            : tensor<10x1x1x1x!qElemType1, {order = #NHWC}> -> tensor<10x1x1x1xsi8, {order = #NHWC}>
        %cast2 = VPU.QuantizeCast(%cst2) {dstElemType = si8}
            : tensor<6x1x1x1x!qElemType3, {order = #NHWC}> -> tensor<6x1x1x1xsi8, {order = #NHWC}>
        %cast3 = VPU.QuantizeCast(%cst3) {dstElemType = si8}
            : tensor<4x1x1x1x!qElemType5, {order = #NHWC}> -> tensor<4x1x1x1xsi8, {order = #NHWC}>

        return %arg, %cast1, %cast2, %cast3 : tensor<4x16xf16>, tensor<10x1x1x1xsi8, {order = #NHWC}>,
            tensor<6x1x1x1xsi8, {order = #NHWC}>, tensor<4x1x1x1xsi8, {order = #NHWC}>
    }

    // CHECK-INIT-FULL: func.func @init
    // CHECK-INIT-FULL:     [[CONCAT:%.+]] = IE.Concat({{%.+}}, {{%.+}}, {{%.+}})
    // CHECK-INIT-FULL-SAME: tensor<4xi8>, tensor<6xi8>, tensor<10xi8>
    // CHECK-INIT-FULL:     return [[CONCAT]]


    // CHECK-MAIN-FULL: func.func @main({{%.+}}: tensor<4x16xf16>, [[BLOB:%.+]]: tensor<20xi8>)
    // CHECK-MAIN-FULL: {{%.+}} = VPU.Slice [[BLOB]] [0] [4] {{.*}} to tensor<4xi8>
    // CHECK-MAIN-FULL: {{%.+}} = VPU.Slice [[BLOB]] [4] [6] {{.*}} to tensor<6xi8>
    // CHECK-MAIN-FULL: {{%.+}} = VPU.Slice [[BLOB]] [10] [10] {{.*}} to tensor<10xi8>


    // CHECK-GEN-ALL: func.func private @init
    // CHECK-GEN-ALL:   [[CONCAT:%.+]] = IE.Concat({{%.+}}, {{%.+}}, {{%.+}})
    // CHECK-GEN-ALL-SAME:  tensor<10xi8>, tensor<6xi8>, tensor<4xi8>
    // CHECK-GEN-ALL:   return [[CONCAT]]

    // CHECK-GEN-ALL: func.func private @main({{%.+}}: tensor<4x16xf16>, [[BLOB:%.+]]: tensor<20xi8>)
    // CHECK-GEN-ALL:   {{%.+}} = VPU.Slice [[BLOB]] [0] [10] {{.*}} to tensor<10xi8>
    // CHECK-GEN-ALL:   {{%.+}} = VPU.Slice [[BLOB]] [10] [6] {{.*}} to tensor<6xi8>
    // CHECK-GEN-ALL:   {{%.+}} = VPU.Slice [[BLOB]] [16] [4] {{.*}} to tensor<4xi8>
}
