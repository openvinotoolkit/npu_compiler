//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIPDPU/utils/utils.hpp"

#include "vpux/compiler/NPU37XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/utils/quantization.hpp"

using namespace vpux;

namespace {

bool isFloatingPointElementType(mlir::Type elementType) {
    return elementType.isFloat() || vpux::isLowFpType(elementType) || vpux::isLowFpTypeQuantized(elementType);
}

}  // namespace

vpux::VPUIPDPU::DpuPvpCounts vpux::VPUIPDPU::computeDpuPvpCounts(ELF::MainOp elfMain, const Logger& log) {
    uint64_t countFPOps = 0;
    uint64_t totalOpCount = 0;

    // After ExpandDPUConfigPass, DPU invariant ops are VPUIPDPU::DPUInvariantOp.
    elfMain->walk([&](VPUIPDPU::DPUInvariantOp invOp) {
        const auto isConv = invOp.getNceTaskType() == vpux::VPUIP::NCETaskType::CONV;

        // Extract input activations, kernel, stride, and output shape from child ops in a single walk.
        int64_t inputChannels = 1;
        mlir::Type inputElementType;
        int64_t kernelH = 1;
        int64_t kernelW = 1;
        int64_t strideX = 0;
        int64_t strideY = 0;
        bool hasStrides = false;
        int64_t outputHeight = 0;
        int64_t outputWidth = 0;
        int64_t outputChannels = 0;
        bool hasOutput = false;
        invOp->walk([&](mlir::Operation* op) {
            llvm::TypeSwitch<mlir::Operation*>(op)
                    .Case<VPUIPDPU::IDUInActivationsOp>([&](auto inActOp) {
                        auto inputNDType = mlir::cast<vpux::NDTypeInterface>(inActOp.getInActivations().getType());
                        inputElementType = inputNDType.getElementType();
                        if (isConv) {
                            inputChannels = inputNDType.getShape()[Dims4D::Act::C];
                        }
                    })
                    .Case<VPUIPDPU::IDUKernelOp>([&](auto kernelOp) {
                        kernelH = kernelOp.getKernelY();
                        kernelW = kernelOp.getKernelX();
                    })
                    .Case<VPUIPDPU::IDUStrideOp>([&](auto strideOp) {
                        strideX = strideOp.getStrideX();
                        strideY = strideOp.getStrideY();
                        hasStrides = true;
                    })
                    .Case<VPUIPDPU::ODUOutTensorSizeOp>([&](auto outSizeOp) {
                        outputHeight = outSizeOp.getDimY();
                        outputWidth = outSizeOp.getDimX();
                        outputChannels = outSizeOp.getDimZ();
                        hasOutput = true;
                    });
        });

        if (hasOutput) {
            const auto opCount = outputChannels * outputHeight * outputWidth * kernelH * kernelW * inputChannels;
            totalOpCount += opCount;

            // PVP throttles ZM convolution where hstride < 3 and vstride < 3.
            if (hasStrides && isConv && (strideX < 3 && strideY < 3)) {
                if (inputElementType && isFloatingPointElementType(inputElementType)) {
                    countFPOps += opCount;
                }
            }
        }
    });

    log.info("Total number of FPOps: {0}", countFPOps);
    log.info("Total number of INTOps: {0}", totalOpCount - countFPOps);
    log.info("Total number of Ops: {0}", totalOpCount);

    return {countFPOps, totalOpCount};
}
