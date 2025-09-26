//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --default-hw-mode-vpu --lower-VPU-to-VPUIP --default-hw-mode-vpuip %s | FileCheck %s --strict-whitespace
// REQUIRES: arch-NPU37XX || arch-NPU40XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK: @InMain
module @InMain {
    net.NetworkInfo entryPoint : @main inputsInfo : {
        DataInfo "input" : tensor<1x512x7x7xf16>
        DataInfo "out_ov_0_hash_12345_concat": tensor<2359296xi8>
    } outputsInfo : {
        DataInfo "output" : tensor<256x512x3x3xf16>
    }

    func.func @main(%arg0: tensor<1x512x7x7xf16, {order = #NHWC}>, %arg1: tensor<2359296xi8>)
            -> tensor<256x512x3x3xf16, {order = #NHWC}> {
        %out = Core.ReinterpretCast(%arg1) : tensor<2359296xi8> -> tensor<256x512x3x3xf16, {order = #NHWC}>
        return %out : tensor<256x512x3x3xf16, {order = #NHWC}>
    }

    // CHECK: func.func @main
    // CHECK-SAME: ({{%.+}}: memref<1x512x7x7xf16, #NHWC, @DDR>, {{%.+}}: memref<2359296xi8, @DDR>, {{%.+}}: memref<256x512x3x3xf16, #NHWC, @DDR>)
    // CHECK-SAME: -> memref<256x512x3x3xf16, #NHWC, @DDR>

    // CHECK-DAG:   [[BLOB:%.+]] = VPURT.DeclareBuffer <NetworkInput> [1] <0> -> memref<{{.+}}x512x3x3xf16, #NHWC, @DDR>
    // CHECK-DAG:   [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<{{.+}}x512x3x3xf16, #NHWC, @DDR>

    // CHECK:       VPUIP.NNDMA {{.*}} inputs([[BLOB]]
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
        DataInfo "output" : tensor<256x512x3x3xf16>
    }

    func.func @main(%arg0: tensor<1x512x7x7xf16, {order = #NHWC}>, %arg1: tensor<51021312xi8>)
            -> tensor<256x512x3x3xf16, {order = #NHWC}> {
        %0 = VPU.Slice %arg1 [48662016] [2359296] : tensor<51021312xi8> to tensor<2359296xi8>
        %out = Core.ReinterpretCast(%0) : tensor<2359296xi8> -> tensor<256x512x3x3xf16, {order = #NHWC}>
        return %out : tensor<256x512x3x3xf16, {order = #NHWC}>
    }

    // CHECK: func.func @main
    // CHECK-SAME: ({{%.+}}: memref<1x512x7x7xf16, #NHWC, @DDR>, {{%.+}}: memref<51021312xi8, @DDR>, {{%.+}}: memref<256x512x3x3xf16, #NHWC, @DDR>)
    // CHECK-SAME: -> memref<256x512x3x3xf16, #NHWC, @DDR>

    // CHECK-DAG:   [[BLOB:%.+]] = VPURT.DeclareBuffer <NetworkInput> [1] <48662016> -> memref<{{.+}}x512x3x3xf16, #NHWC, @DDR>
    // CHECK-DAG:   [[OUT:%.+]] = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<{{.+}}x512x3x3xf16, #NHWC, @DDR>

    // CHECK:       VPUIP.NNDMA {{.*}} inputs([[BLOB]]
    // CHECK-SAME:      outputs([[OUT]]
}
