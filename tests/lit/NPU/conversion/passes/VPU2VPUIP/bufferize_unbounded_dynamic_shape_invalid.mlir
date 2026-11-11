//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// RUN: vpux-opt --verify-diagnostics --init-compiler="platform=%platform%" --one-shot-bufferize-VPU-to-VPUIP %s
// REQUIRES: platform-NPU3720 || platform-NPU4000 || platform-NPU5010 || platform-NPU6010

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// expected-error-re@+1 {{Cannot allocate a buffer of type 'memref<1x1x1x?xf16, [@CMX_NN, 0]>' at '{{.+}}': its shape has a dimension without an upper bound, which is supported only in DDR}}
module @UnboundedCMXBuffer {
    func.func @main(%input: tensor<1x1x1x?xf16>) -> tensor<1x1x1x?xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}> {
        %copy = VPU.Copy(%input) {out_mem_space = [@CMX_NN, 0]} : tensor<1x1x1x?xf16> -> tensor<1x1x1x?xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
        return %copy : tensor<1x1x1x?xf16, {mem_space = [@CMX_NN, 0], order = #NCHW}>
    }
}
