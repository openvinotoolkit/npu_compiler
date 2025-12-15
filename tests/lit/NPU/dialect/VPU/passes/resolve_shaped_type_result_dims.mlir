//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --resolve-shaped-type-result-dims %s | FileCheck %s
// REQUIRES: arch-NPU37XX || arch-NPU40XX || arch-NPU50XX

// -----
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @Conv
func.func @Conv(%arg0: tensor<1x16x?x8xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 8]> : tensor<4xsi64>, order = #NHWC}>, %arg1: tensor<1x16x4x8xf16, {order = #NHWC}>) -> (tensor<1x32x?x4xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 50, 4]> : tensor<4xsi64>, order = #NHWC}>, index) {
    // CHECK: [[IN:%.+]]: tensor<1x16x?x8xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 8]> : tensor<4xsi64>, order = #NHWC}>
    %C2 = arith.constant 2 : index
    // CHECK: [[C2:%.+]] = arith.constant 2 : index
    %cst_0 = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    // CHECK: [[WEIGHT:%.+]] = const.Declare
    %CONV = VPU.NCE.Convolution(%arg0, %cst_0) {ppe = #VPU.PPEStub<>, pad = #VPU.Padding<left = 1 : i64, right = 0 : i64, top = 1 : i64, bottom = 0 : i64>, rawFilterShape = [32, 16, 3, 3], strides = [2, 2]} : tensor<1x16x?x8xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 100, 8]> : tensor<4xsi64>, order = #NHWC}>, tensor<32x16x3x3xf16, {order = #NHWC}> -> tensor<1x32x?x4xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 50, 4]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[CONV:%.+]] = VPU.NCE.Convolution([[IN]], [[WEIGHT]])
    %DIM = tensor.dim %CONV, %C2 : tensor<1x32x?x4xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 50, 4]> : tensor<4xsi64>, order = #NHWC}>
    // CHECK: [[VAL:%.+]] = tensor.dim [[IN]], [[C2]]
    // CHECK: [[DIM:%.+]] = arith.divsi [[VAL]], [[C2]] : index
    return %CONV, %DIM : tensor<1x32x?x4xf16, {bounds = #const.OpaqueI64Elements<[1, 32, 50, 4]> : tensor<4xsi64>, order = #NHWC}>, index
    // CHECK: return [[CONV]], [[DIM]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @Eltwise
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x?x256xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 340, 256]> : tensor<4xsi64>, order = #NHWC}>
func.func @Eltwise(%arg0: tensor<1x16x?x256xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 340, 256]> : tensor<4xsi64>, order = #NHWC}>) -> (tensor<1x16x?x256xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 340, 256]> : tensor<4xsi64>, order = #NHWC}>, index) {
    %C2 = arith.constant 2 : index
    %ELTWISE = VPU.NCE.Eltwise(%arg0, %arg0) {
            input_padding = [0, 0, 0, 0],
            op_type = #VPU.eltwise_type<SUBTRACT>,
            multiClusterStrategy = #VPU.multi_cluster_strategy<SplitOverHeight>,
            ppe = #VPU.PPEStub<>,
            tilingStrategy = [1, 1, 2, 1]
        } -> tensor<1x16x?x256xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 340, 256]> : tensor<4xsi64>, order = #NHWC}>
    %DIM = tensor.dim %ELTWISE, %C2 : tensor<1x16x?x256xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 340, 256]> : tensor<4xsi64>, order = #NHWC}>
    return %ELTWISE,%DIM : tensor<1x16x?x256xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 340, 256]> : tensor<4xsi64>, order = #NHWC}>, index

    // CHECK:       [[C2:%.+]] = arith.constant 2 : index
    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.Eltwise([[INPUT]], [[INPUT]])
    // CHECK:       [[DIM:%.+]] = tensor.dim [[INPUT]], [[C2]]
    // CHECK:       return [[OUTPUT]], [[DIM]]
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NCEMaxPool
// CHECK-SAME:  [[INPUT0:%arg[0-9]]]: tensor<1x16x?x15xf16, {bounds = #const.OpaqueI64Elements<[1, 16, 340, 15]> : tensor<4xsi64>, mem_space = @CMX_NN, order = #NHWC}>,
// CHECK-SAME:  [[INPUT1:%arg[0-9]]]: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
func.func @NCEMaxPool(%arg0: tensor<1x16x?x15xf16, {mem_space = @CMX_NN, bounds = #const.OpaqueI64Elements<[1, 16, 340, 15]> : tensor<4xsi64>, order = #NHWC}>,
                 %arg1: tensor<16x1x1x4xsi32, {mem_space = @CMX_NN, order = #NHWC}>
                 ) -> (tensor<1x16x?x15xf16, {mem_space = @CMX_NN, bounds = #const.OpaqueI64Elements<[1, 16, 340, 15]> : tensor<4xsi64>, order = #NCHW}>, index) {
    %C2 = arith.constant 2 : index
    %MAXPOOL = VPU.NCE.MaxPool(%arg0, %arg1) {
        kernel_size = [1, 1],
        pad = #VPU.Padding<left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 0 : i64>,
        ppe = #VPU.PPEStub<>,
        strides = [1, 1]
    } -> tensor<1x16x?x15xf16, {mem_space = @CMX_NN, bounds = #const.OpaqueI64Elements<[1, 16, 340, 15]> : tensor<4xsi64>, order = #NCHW}> {
        VPU.DPU.Workload outOffsets [0, 0, 0, 0] outSizes [1, 16, 15, 15] <left = 0 : i64, right = 0 : i64, top = 0 : i64, bottom = 13 : i64> #VPU.mpe_mode<CUBOID_16x16>
    }
    %DIM = tensor.dim %arg0, %C2 : tensor<1x16x?x15xf16, {mem_space = @CMX_NN, bounds = #const.OpaqueI64Elements<[1, 16, 340, 15]> : tensor<4xsi64>, order = #NHWC}>
    return %MAXPOOL,%DIM : tensor<1x16x?x15xf16, {mem_space = @CMX_NN, bounds = #const.OpaqueI64Elements<[1, 16, 340, 15]> : tensor<4xsi64>, order = #NCHW}>, index

    // CHECK:       [[C2:%.+]] = arith.constant 2 : index
    // CHECK:       [[OUTPUT:%.+]] = VPU.NCE.MaxPool([[INPUT0]], [[INPUT1]] )
    // CHECK:       [[DIM:%.+]] = tensor.dim [[INPUT0]], [[C2]]
    // CHECK:       return [[OUTPUT]], [[DIM]]
}
