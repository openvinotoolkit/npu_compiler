//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/utils/compression_utils.hpp"
#include "vpux/compiler/utils/memref_attr_utils.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_ADJUSTSPILLSIZE
#define GEN_PASS_DEF_ADJUSTSPILLSIZE
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

class AdjustSpillSizePass final : public VPUIP::impl::AdjustSpillSizeBase<AdjustSpillSizePass> {
public:
    explicit AdjustSpillSizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    int64_t getAdjustedSpillBufferSize(vpux::NDTypeInterface origTypeThatGotSpilled);
    void updateSpillWrite(VPUIP::NNDMAOp dmaOp);
    void updateSpillRead(VPUIP::NNDMAOp dmaOp);

    mlir::DenseMap<int64_t, mlir::Type> _spillIdAndTypeMap;
};

int64_t AdjustSpillSizePass::getAdjustedSpillBufferSize(vpux::NDTypeInterface origTypeThatGotSpilled) {
    int64_t numberOfDmas = 1;
    // In case of segmented buffer each chunk needs to satisfy
    // compression requirements as each will be handled by dedicated compress DMA
    if (auto distributedType = origTypeThatGotSpilled.dyn_cast<VPUIP::DistributedBufferType>()) {
        const auto distributionAttr = distributedType.getDistribution();
        const auto distributionMode = distributionAttr.getMode().getValue();

        if (distributionMode == VPU::DistributionMode::SEGMENTED ||
            distributionMode == VPU::DistributionMode::OVERLAPPED) {
            numberOfDmas = distributionAttr.getNumClusters().getInt();
        }
    }

    // In worst case scenario depending on the content of activation, its final size after
    // compression might be bigger than original size. Compiler before performing DDR
    // allocation needs to adjust required size by this buffer
    return numberOfDmas * updateSizeForCompression(origTypeThatGotSpilled.getTotalAllocSize().count());
}

// Update spill-write op have new type that reflects desired size of allocation
void AdjustSpillSizePass::updateSpillWrite(VPUIP::NNDMAOp dmaOp) {
    _log.trace("Spill Write op identified - '{0}'", dmaOp->getLoc());

    auto spillBuffer = dmaOp.getOutputBuff();

    auto asyncOp = dmaOp->getParentOfType<mlir::async::ExecuteOp>();
    auto clusterOp = dmaOp->getParentOfType<VPUIP::NCEClusterTilingOp>();
    mlir::BlockArgument blockArg;
    mlir::Type spillSourceBufferType;
    if (clusterOp) {
        asyncOp = clusterOp->getParentOfType<mlir::async::ExecuteOp>();
        blockArg = dmaOp.getOutputBuff().cast<mlir::BlockArgument>();
        spillBuffer = clusterOp->getOperand(blockArg.getArgNumber());
        spillSourceBufferType =
                clusterOp->getOperand(dmaOp.getInput().cast<mlir::BlockArgument>().getArgNumber()).getType();
    } else {
        spillSourceBufferType = dmaOp.getInput().getType();
    }
    VPUX_THROW_WHEN(asyncOp == nullptr, "No async execute identified for given SpillWrite DmaOp - '{0}'",
                    dmaOp->getLoc());

    auto spillAllocOp = spillBuffer.getDefiningOp();
    VPUX_THROW_UNLESS((mlir::isa<mlir::memref::AllocOp, VPURT::Alloc>(spillAllocOp)),
                      "Unexpected allocation operation for spill buffer - '{0}'", spillAllocOp);

    auto spillAllocOpResult = spillAllocOp->getResult(0);

    // Allocation size for activation spill should be adjusted. This is because later those spills will be converted
    // to compressed DMA and in worst case scenario size of compressed activation can be in fact bigger than
    // original buffer. To prevent from memory corruption spill buffer size needs to be adjusted

    auto spillType = spillAllocOpResult.getType().cast<vpux::NDTypeInterface>();

    auto newSpillType = vpux::setAllocSizeAttr(spillType, getAdjustedSpillBufferSize(spillSourceBufferType));
    newSpillType = vpux::setCompressionState(newSpillType, VPUIP::CompressionState::CompressionCandidate);

    _log.nest().trace("New spilled buffer type - '{0}'", newSpillType);

    // Update type of allocation operation and also result types for spill write operation
    spillAllocOpResult.setType(newSpillType);
    dmaOp->getResult(0).setType(newSpillType);

    if (clusterOp) {
        blockArg.setType(newSpillType);
        clusterOp->getResult(0).setType(newSpillType);
    }

    auto asyncSpillResult = asyncOp.getBodyResults()[0];
    asyncSpillResult.setType(mlir::async::ValueType::get(newSpillType));

    _spillIdAndTypeMap[dmaOp.getSpillId().value()] = newSpillType;
    dmaOp.setCompressCandidate(true);
}

// Based on corresponding spill-write op this function will update
// spill-read operation to have new type that reflects desired size of allocation
void AdjustSpillSizePass::updateSpillRead(VPUIP::NNDMAOp dmaOp) {
    _log.trace("Spill Read op identified - '{0}'", dmaOp->getLoc());

    auto spillId = dmaOp.getSpillId().value();
    VPUX_THROW_UNLESS(_spillIdAndTypeMap.find(spillId) != _spillIdAndTypeMap.end(),
                      "No matching spill write was located before");

    auto newSpillTypeMemref = _spillIdAndTypeMap[spillId];

    auto spillBuffer = dmaOp.getInput();

    // Update type block args of operations which wrap spill read NNDMA op
    auto asyncOp = dmaOp->getParentOfType<mlir::async::ExecuteOp>();
    auto clusterOp = dmaOp->getParentOfType<VPUIP::NCEClusterTilingOp>();

    if (clusterOp) {
        asyncOp = clusterOp->getParentOfType<mlir::async::ExecuteOp>();
        auto blockArg = dmaOp.getInput().dyn_cast<mlir::BlockArgument>();

        spillBuffer = clusterOp->getOperand(blockArg.getArgNumber());

        blockArg.setType(newSpillTypeMemref);
    }

    VPUX_THROW_WHEN(asyncOp == nullptr, "No async execute identified for given SpillRead DmaOp - '{0}'",
                    dmaOp->getLoc());

    if (auto blockArg = spillBuffer.dyn_cast<mlir::BlockArgument>()) {
        blockArg.setType(newSpillTypeMemref);
    }
    dmaOp.setCompressCandidate(true);
}

void AdjustSpillSizePass::safeRunOnFunc() {
    auto func = getOperation();

    func->walk([&](VPUIP::NNDMAOp dmaOp) {
        if (!dmaOp.getSpillId().has_value()) {
            return;
        }

        const auto inType = dmaOp.getInput().getType().cast<vpux::NDTypeInterface>();
        const auto outType = dmaOp.getOutput().getType().cast<vpux::NDTypeInterface>();

        if (inType.getMemoryKind() == VPU::MemoryKind::CMX_NN && outType.getMemoryKind() == VPU::MemoryKind::DDR) {
            if (isSupportedBufferSizeForCompression(inType)) {
                updateSpillWrite(dmaOp);
            }
        } else if (inType.getMemoryKind() == VPU::MemoryKind::DDR &&
                   outType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
            if (isSupportedBufferSizeForCompression(outType)) {
                updateSpillRead(dmaOp);
            }
        }
    });
}

}  // namespace

//
// createAdjustSpillSizePass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createAdjustSpillSizePass(Logger log) {
    return std::make_unique<AdjustSpillSizePass>(log);
}
