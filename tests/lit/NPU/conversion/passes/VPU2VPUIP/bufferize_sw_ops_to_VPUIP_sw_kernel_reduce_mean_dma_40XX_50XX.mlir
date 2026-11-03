//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="platform=%platform% allow-custom-values=true" --one-shot-bufferize-VPU-to-VPUIP %s | FileCheck %s
// REQUIRES: platform-NPU4000 || platform-NPU5010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK: func.func nested @builtin_ReduceMeanDMA(
// CHECK-SAME: memref<*xf16, [@CMX_NN, 0]>
// CHECK-SAME: memref<*xui8, [@CMX_NN, 0]>
// CHECK-SAME: memref<*xf16, [@CMX_NN, 0]>
// CHECK-SAME: memref<*xui8, [@CMX_NN, 0]>
// CHECK-SAME: i64
// CHECK-SAME: ) attributes {VPU.kernel_code = "reduce_mean_dma.cpp", VPU.kernel_entry = "reduce_mean_dma", VPU.kernel_name = "reduce_mean_dma", VPU.task_type = @COMPUTE}

// CHECK-LABEL: func.func @ReduceMeanDMA(
// CHECK-SAME: [[INPUT:%.+]]: memref<1x7x?x?xf16{{.+}}, [[AUX_IN:%.+]]: memref<65536xui8>
module attributes {config.platform = #config.platform<NPU5010>} {
func.func @ReduceMeanDMA(%input: tensor<1x7x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 7, 13, 19]> : tensor<4xsi64>, order = #NCHW}>,
                         %aux: tensor<65536xui8>) -> tensor<1x7x1x1xf16> {
    %in = VPU.Copy(%input) {out_mem_space = [@CMX_NN, 0]}
        : tensor<1x7x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 7, 13, 19]> : tensor<4xsi64>, order = #NCHW}>
        -> tensor<1x7x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 7, 13, 19]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}>
    %in_aux = VPU.Copy(%aux) {out_mem_space = [@CMX_NN, 0]}
        : tensor<65536xui8> -> tensor<65536xui8, {mem_space = [@CMX_NN, 0], order = affine_map<(d0) -> (d0)>}>

    %out_cmx = VPU.ReduceMeanDMA(%in, %in_aux)
        : tensor<1x7x?x?xf16, {bounds = #const.OpaqueI64Elements<[1, 7, 13, 19]> : tensor<4xsi64>, mem_space = [@CMX_NN, 0], order = #NCHW}>,
          tensor<65536xui8, {mem_space = [@CMX_NN, 0], order = affine_map<(d0) -> (d0)>}>
        -> tensor<1x7x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>

    %out = VPU.Copy(%out_cmx) : tensor<1x7x1x1xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}> -> tensor<1x7x1x1xf16>
    return %out : tensor<1x7x1x1xf16>

    // CHECK: [[IN_CMX:%.+]] = VPUIP.Copy inputs([[INPUT]] : memref<1x7x?x?xf16
    // CHECK: [[AUX_CMX:%.+]] = VPUIP.Copy inputs([[AUX_IN]] : memref<65536xui8>)
    // CHECK: [[OUT_CMX:%.+]] = memref.alloc() : memref<1x7x1x1xf16, [@CMX_NN, 0]>
    // CHECK: [[KERNEL_OUT:%.+]]:2 = VPUIP.SW.Kernel
    // CHECK-SAME: @VPU.SW::@builtin_ReduceMeanDMA
    // CHECK-SAME: inputs([[IN_CMX]] as [[ARG_IN:%arg[0-9]+]]: memref<1x7x?x?xf16{{[^)]+}}, [[AUX_CMX]] as [[ARG_AUX_IN:%arg[0-9]+]]: memref<65536xui8, [@CMX_NN, 0]>)
    // CHECK-SAME: outputs([[OUT_CMX]] as [[ARG_OUT:%arg[0-9]+]]: memref<1x7x1x1xf16, [@CMX_NN, 0]>, [[AUX_CMX]] as [[ARG_AUX_OUT:%arg[0-9]+]]: memref<65536xui8, [@CMX_NN, 0]>)
    // CHECK: VPUIP.SW.Kernel.run {attrs = [8589934593]}([[ARG_IN]], [[ARG_AUX_IN]], [[ARG_OUT]], [[ARG_AUX_OUT]])
}
}
