//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform%" --default-hw-mode-vpu --lower-VPU-to-VPUIP --default-hw-mode-vpuip %s | FileCheck %s --strict-whitespace
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: @InMain
module @InMain {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x512x7x7xf16>
        DataInfo "out_ov_0_hash_12345_concat": tensor<1048576xi8>
    } outputsInfo : {
        DataInfo "output" : tensor<64x512x4x4xf16>
    }

    func.func @main(%arg0: tensor<1x512x7x7xf16, {order = #NHWC}>, %arg1: tensor<1048576xi8>)
            -> tensor<64x512x4x4xf16, {order = #NHWC}> {
        %out = Core.ReinterpretCast(%arg1) : tensor<1048576xi8> -> tensor<64x512x4x4xf16, {order = #NHWC}>
        return %out : tensor<64x512x4x4xf16, {order = #NHWC}>
    }

    // CHECK: func.func @main
    // CHECK-SAME: ({{%.+}}: memref<1x512x7x7xf16, {order = #NHWC}, @DDR>, {{%.+}}: memref<1048576xi8, @DDR>, {{%.+}}: memref<64x512x4x4xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME: -> memref<64x512x4x4xf16, {order = #NHWC}, @DDR>

    // CHECK-DAG:   [[BLOB:%.+]] = VPURT.DeclareBuffer <NetworkInput> [1] <0> -> memref<{{.+}}x512x4x4xf16, {order = #NHWC}, @DDR>
    // CHECK-DAG:   [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<{{.+}}x512x4x4xf16, {order = #NHWC}, @DDR>

    // CHECK:       VPUIP.NNDMA <{{.+}}> inputs([[BLOB]]
    // CHECK-SAME:      outputs([[OUT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: @InMainWithSlice
module @InMainWithSlice {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x512x7x7xf16>
        DataInfo "out_ov_0_hash_12345_concat": tensor<51021312xi8>
    } outputsInfo : {
        DataInfo "output" : tensor<64x512x4x4xf16>
    }

    func.func @main(%arg0: tensor<1x512x7x7xf16, {order = #NHWC}>, %arg1: tensor<51021312xi8>)
            -> tensor<64x512x4x4xf16, {order = #NHWC}> {
        %0 = VPU.Slice %arg1 [48662016] [1048576] : tensor<51021312xi8> to tensor<1048576xi8>
        %out = Core.ReinterpretCast(%0) : tensor<1048576xi8> -> tensor<64x512x4x4xf16, {order = #NHWC}>
        return %out : tensor<64x512x4x4xf16, {order = #NHWC}>
    }

    // CHECK: func.func @main
    // CHECK-SAME: ({{%.+}}: memref<1x512x7x7xf16, {order = #NHWC}, @DDR>, {{%.+}}: memref<51021312xi8, @DDR>, {{%.+}}: memref<64x512x4x4xf16, {order = #NHWC}, @DDR>)
    // CHECK-SAME: -> memref<64x512x4x4xf16, {order = #NHWC}, @DDR>

    // The reinterpreted slice is below the 2 MiB preemption budget. Some platforms still split for scheduling,
    // so only assert that at least one matching DMA chunk is produced from the sliced input region.
    // CHECK-DAG:   [[BLOB:%.+]] = VPURT.DeclareBuffer <NetworkInput> [1] <48662016> -> memref<{{.+}}x512x4x4xf16, {order = #NHWC}, @DDR>
    // CHECK-DAG:   [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<{{.+}}x512x4x4xf16, {order = #NHWC}, @DDR>

    // CHECK:       VPUIP.NNDMA <{{.+}}> inputs([[BLOB]]
    // CHECK-SAME:      outputs([[OUT]]
}
