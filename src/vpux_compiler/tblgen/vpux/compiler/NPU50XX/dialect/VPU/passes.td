//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU_ARCH_50XX_PASSES
#define VPUX_COMPILER_DIALECT_VPU_ARCH_50XX_PASSES

include "mlir/Pass/PassBase.td"

//
// DecomposeSoftmaxInSdpa
//

def DecomposeSoftmaxInSdpa : PassBase<"decompose-softmax-in-sdpa", "vpux::FunctionPass"> {
    let summary = "Decompose softmax and offload parts of it onto DPU";

    let description = [{
        Re-balance the workloads between DPU and Shave for SDPA patterns
        Transform 
            Operand -> Softmax -> Matmul(Convolution)
        to
            Operand ->
                    \                 \
            ReduceMax(fp16) -> Subtract(fp16) -> Convert(fp32) -> Exp(fp32) \
            -> Convert(fp16) ->
                    \   NCEReduceSum -> Divide(Inverse) -> NCEConvolution(Expand)
                    \   NCEConvolution, scaling up by 1/rescaler    \
                            \ --------------------------------NCEEltwise.Multiply, scaling down by rescaler
        
        TODO:
            Replace `ReduceMax(fp16) -> Subtract(fp16) -> Convert(fp32) -> Exp(fp32) -> Convert(fp16)`
                with a new ShaveOp Exp_stabilized. The ReduceMax has a chance to be fused into the previous MatMul.
            Replace `NCEReduceSum -> Divide(Inverse)` with NCEReduceSum(ppe.mode=<INV>)
    }];

    let constructor = "vpux::VPU::arch50xx::createDecomposeSoftmaxInSdpaPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

//
// AutopadChannels
//

def AutopadChannels : PassBase<"autopad-channels", "vpux::FunctionPass"> {
    let summary = "Autopad channels";

    let description = [{
        Enable autopad for NCE operations where this feature could improve performance.
        Autopadding allows NCE operations to consume and produce data where the number of channels is smaller than 16.

        At the moment, only ODU autopad is supported by this pass. It searches for NCE operations whose results have
        the channels padded to 16 and is consumed by Shave operations, then removes the padding.
    }];


    let constructor = "vpux::VPU::arch50xx::createAutopadChannelsPass()";

    let dependentDialects = [
        "vpux::VPU::VPUDialect"
    ];
}

#endif
