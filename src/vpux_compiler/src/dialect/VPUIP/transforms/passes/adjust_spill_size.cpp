//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/compression_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/memref_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/compression_utils.hpp"

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
    if (auto distributedType = mlir::dyn_cast<vpux::VPUIP::DistributedBufferType>(origTypeThatGotSpilled)) {
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
    mlir::Type spillSourceBufferType = dmaOp.getInput().getType();
    VPUX_THROW_WHEN(asyncOp == nullptr, "No async execute identified for given SpillWrite DmaOp - '{0}'",
                    dmaOp->getLoc());

    auto spillAllocOp = spillBuffer.getDefiningOp();
    VPUX_THROW_UNLESS((mlir::isa<mlir::memref::AllocOp, VPURT::Alloc>(spillAllocOp)),
                      "Unexpected allocation operation for spill buffer - '{0}'", spillAllocOp);

    auto spillAllocOpResult = spillAllocOp->getResult(0);

    // Allocation size for activation spill should be adjusted. This is because later those spills will be converted
    // to compressed DMA and in worst case scenario size of compressed activation can be in fact bigger than
    // original buffer. To prevent from memory corruption spill buffer size needs to be adjusted

    auto spillType = mlir::cast<vpux::NDTypeInterface>(spillAllocOpResult.getType());

    auto newSpillType = VPUIP::setAllocSizeAttr(spillType, getAdjustedSpillBufferSize(spillSourceBufferType));
    newSpillType = VPUIP::setCompressionState(newSpillType, VPUIP::CompressionState::CompressionCandidate);

    _log.nest().trace("New spilled buffer type - '{0}'", newSpillType);

    // Update type of allocation operation and also result types for spill write operation
    spillAllocOpResult.setType(newSpillType);
    dmaOp->getResult(0).setType(newSpillType);

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
    VPUX_THROW_WHEN(_spillIdAndTypeMap.find(spillId) == _spillIdAndTypeMap.end(),
                    "No matching spill write was located before");

    auto newSpillTypeMemref = _spillIdAndTypeMap[spillId];

    auto spillBuffer = dmaOp.getInput();

    // Update type block args of operations which wrap spill read NNDMA op
    auto asyncOp = dmaOp->getParentOfType<mlir::async::ExecuteOp>();

    VPUX_THROW_WHEN(asyncOp == nullptr, "No async execute identified for given SpillRead DmaOp - '{0}'",
                    dmaOp->getLoc());

    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(spillBuffer)) {
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

        const auto inType = mlir::cast<vpux::NDTypeInterface>(dmaOp.getInput().getType());
        const auto outType = mlir::cast<vpux::NDTypeInterface>(dmaOp.getOutput().getType());

        if (inType.getMemoryKind() == VPU::MemoryKind::CMX_NN && outType.getMemoryKind() == VPU::MemoryKind::DDR) {
            if (VPUIP::isSupportedBufferSizeForCompression(inType)) {
                updateSpillWrite(dmaOp);
            }
        } else if (inType.getMemoryKind() == VPU::MemoryKind::DDR &&
                   outType.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
            if (VPUIP::isSupportedBufferSizeForCompression(outType)) {
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
