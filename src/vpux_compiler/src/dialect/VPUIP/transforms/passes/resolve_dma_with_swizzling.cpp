//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_RESOLVEDMAWITHSWIZZLING
#define GEN_PASS_DEF_RESOLVEDMAWITHSWIZZLING
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// ResolveDMAWithSwizzlingPass
//

class ResolveDMAWithSwizzlingPass final : public VPUIP::impl::ResolveDMAWithSwizzlingBase<ResolveDMAWithSwizzlingPass> {
public:
    explicit ResolveDMAWithSwizzlingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void createNewFlatAlignedConst(Const::DeclareOp& constOp) {
    const auto& contentAttr = constOp.getContentAttr();

    const auto newOutputType = contentAttr.getType();
    auto constType = constOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    constType = mlir::isa<VPUIP::DistributedBufferType>(constType)
                        ? VPU::changeShapeElemTypeForDuplicatedDistributedBuffers(constType, newOutputType.getShape(),
                                                                                  newOutputType.getElementType())
                        : constType.changeShapeElemType(newOutputType.getShape(), newOutputType.getElementType());

    constOp.getOutput().setType(constType);
}

void ResolveDMAWithSwizzlingPass::safeRunOnFunc() {
    auto func = getOperation();

    func->walk([&](VPURT::TaskOp taskOp) {
        if (taskOp.getExecutorKind() != VPU::ExecutorKind::DMA_NN) {
            return;
        }

        auto dmaOp = mlir::dyn_cast<VPUIP::NNDMAOp>(taskOp.getInnerTaskOp());
        if (dmaOp == nullptr) {
            return;
        }

        auto inputOp = dmaOp.getInput().getDefiningOp();
        auto outputBuff = dmaOp.getOutputBuff().getDefiningOp<VPURT::DeclareBufferOp>();

        if (inputOp == nullptr || outputBuff == nullptr) {
            return;
        }

        const auto inputBuffType = dmaOp.getInput().getType().cast<vpux::NDTypeInterface>();
        const auto outputBuffType = dmaOp.getOutputBuff().getType().cast<vpux::NDTypeInterface>();

        auto inputSwizzling = getSwizzlingKey(inputBuffType);
        auto outputSwizzling = getSwizzlingKey(outputBuffType);

        VPUX_THROW_WHEN(inputSwizzling != outputSwizzling, "Incompatible swizzling setting on task - '{0}'", taskOp);

        if (inputSwizzling == 0) {
            return;
        }

        // If total allocation size is same as the one resulting from buffer shape and strides
        // then no modification is needed as DMA src/dst represent total swizzled buffer
        const auto memShape = inputBuffType.getMemShape();
        const auto memStrides = inputBuffType.getMemStrides();

        const auto outputMemShape = outputBuffType.getMemShape();
        const auto outputMemStrides = outputBuffType.getMemStrides();

        auto buffAllocSize = inputBuffType.getTotalAllocSize().count();
        const auto outputBuffAllocSize = outputBuffType.getTotalAllocSize().count();
        const auto buffRealSize = Byte(memStrides.front() * memShape.front()).count();
        const auto outputBuffRealSize = Byte(outputMemStrides.front() * outputMemShape.front()).count();

        auto inputAligned = (buffRealSize == buffAllocSize);
        auto outputAligned = (outputBuffRealSize == outputBuffAllocSize);

        if (dmaOp.getCompressCandidateAttr() != nullptr) {
            if (inputBuffType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
                // Potential activation compression. Only input needs to have size aligned
                // Assume output is already aligned
                outputAligned = true;
            } else {
                // Potential activation decompression. Only output needs to have size aligned.
                // Assume input is already aligned
                inputAligned = true;
                // In case of DMA DDR2CMX (spill-read decompress DMA) for
                // preparing new tensor dimensions for swizzled buffer use buffer type in CMX (output buffer)
                // instead of source buffer type in DDR (input buffer) as it might have size adjusted for
                // worst case compressed size
                buffAllocSize = outputBuffAllocSize;
            }
        }

        if (inputAligned && outputAligned) {
            _log.trace("DMA already represent total size of swizzled buffer, dmaOp - '{0}'", taskOp->getLoc());
            return;
        }

        _log.trace("Identified DMA task with swizzled buffers that require conversion to flat tensor, dmaOp - '{0}', "
                   "real size - '{1}', allocation size '{2}'",
                   taskOp->getLoc(), buffRealSize, buffAllocSize);

        auto newInputBuffType = inputBuffType;
        auto newOutputBuffType = outputBuffType;

        // Create new shape in the form of flat tensor that will represent total swizzled buffer
        // of size that is explicitly aligned to 512 as required by HW
        SmallVector<int64_t> newShape(memShape.size(), 1);

        if (inputBuffType.getElemTypeSize().count() == 1) {
            newShape[0] = buffAllocSize * CHAR_BIT;
        } else if (inputBuffType.getElementType().isF16()) {
            newShape[0] = buffAllocSize / static_cast<int64_t>(sizeof(vpux::type::float16));
        } else {
            newShape[0] = buffAllocSize;
            auto newElementType = getUInt8Type(taskOp->getContext());
            newInputBuffType = newInputBuffType.changeElemType(newElementType);
            newOutputBuffType = newOutputBuffType.changeElemType(newElementType);
        }

        newInputBuffType =
                mlir::isa<VPUIP::DistributedBufferType>(newInputBuffType)
                        ? VPU::changeShapeElemTypeForDuplicatedDistributedBuffers(newInputBuffType, ShapeRef(newShape),
                                                                                  newInputBuffType.getElementType())
                        : newInputBuffType.changeShapeElemType(ShapeRef(newShape), newInputBuffType.getElementType());
        newOutputBuffType =
                mlir::isa<VPUIP::DistributedBufferType>(newOutputBuffType)
                        ? VPU::changeShapeElemTypeForDuplicatedDistributedBuffers(newOutputBuffType, ShapeRef(newShape),
                                                                                  newOutputBuffType.getElementType())
                        : newOutputBuffType.changeShapeElemType(ShapeRef(newShape), newOutputBuffType.getElementType());

        mlir::OpBuilder builder(taskOp.getOperation());

        mlir::Operation* newInputOp;

        builder.setInsertionPoint(inputOp);
        if (auto inputBuff = dmaOp.getInput().getDefiningOp<VPURT::DeclareBufferOp>()) {
            // Create new source flat buffer with aligned size
            auto newInputBuff = builder.create<VPURT::DeclareBufferOp>(
                    appendLoc(inputBuff->getLoc(), "_flat_buffer_alloc"), newInputBuffType, inputBuff.getSectionAttr(),
                    inputBuff.getSectionIndexAttr(), inputBuff.getByteOffsetAttr(), inputBuff.getSwizzlingKeyAttr());
            _log.nest().trace("Create new source flat buffer allocation of shape - '{0}', op - '{1}'",
                              ShapeRef(newShape), newInputBuff->getLoc());
            newInputOp = newInputBuff.getOperation();
        } else if (auto inputCst = dmaOp.getInput().getDefiningOp<Const::DeclareOp>()) {
            // Create new constant with flat shape with aligned size
            createNewFlatAlignedConst(inputCst);
            _log.nest().trace("Make constant of flat aligned shape - '{0}', op - '{1}'", ShapeRef(newShape),
                              inputCst->getLoc());
            newInputOp = inputCst.getOperation();
        } else {
            VPUX_THROW("Unsupported swizzled source operand of DMA '{0}'", inputOp->getName());
        }

        // Create new destination flat buffer
        builder.setInsertionPoint(outputBuff);
        auto newOutputBuff = builder.create<VPURT::DeclareBufferOp>(
                appendLoc(outputBuff->getLoc(), "_flat_buffer_alloc"), newOutputBuffType, outputBuff.getSectionAttr(),
                outputBuff.getSectionIndexAttr(), outputBuff.getByteOffsetAttr(), outputBuff.getSwizzlingKeyAttr());
        _log.nest().trace("Create new destination flat buffer allocation of shape - '{0}', op - '{1}'",
                          ShapeRef(newShape), newOutputBuff->getLoc());

        builder.setInsertionPoint(dmaOp);
        auto newDmaOp = builder.create<VPUIP::NNDMAOp>(
                appendLoc(dmaOp->getLoc(), "_flat_buffer_dma"), newInputOp->getResult(0), newOutputBuff,
                dmaOp.getPortAttr(), dmaOp.getIsOutOfOrderAttr(), dmaOp.getIsCriticalAttr(), dmaOp.getSpillIdAttr(),
                dmaOp.getCompressCandidateAttr(), /*dmaHwpId=*/nullptr,
                /*profilingMetadata=*/nullptr, dmaOp.getSplitCandidateAttr(), /*profiling_buffer_mgmt=*/nullptr,
                dmaOp.getFusionIdAttr());
        _log.nest().trace("Create new DMA - '{0}'", newDmaOp->getLoc());

        dmaOp->erase();
        if (inputOp->getUsers().empty()) {
            inputOp->erase();
        }
        if (outputBuff->getUsers().empty()) {
            outputBuff->erase();
        }
    });
}

}  // namespace

//
// createResolveDMAWithSwizzlingPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createResolveDMAWithSwizzlingPass(Logger log) {
    return std::make_unique<ResolveDMAWithSwizzlingPass>(log);
}
