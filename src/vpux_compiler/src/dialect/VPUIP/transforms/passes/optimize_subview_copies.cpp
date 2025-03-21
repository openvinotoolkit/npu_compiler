//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_OPTIMIZESUBVIEWCOPIES
#define GEN_PASS_DEF_OPTIMIZESUBVIEWCOPIES
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// OptimizeSubviewCopiesPass
//

class OptimizeSubviewCopiesPass final : public VPUIP::impl::OptimizeSubviewCopiesBase<OptimizeSubviewCopiesPass> {
public:
    struct SubviewSubgraphInfo {
        Byte maxRequiredCMX;
        SmallVector<VPUIP::SubViewOp> subviews;

        SubviewSubgraphInfo(): maxRequiredCMX(Byte(0)){};
        SubviewSubgraphInfo(const Byte& maxCMX, ArrayRef<VPUIP::SubViewOp> subviewOps)
                : maxRequiredCMX(maxCMX), subviews(subviewOps){};

        ~SubviewSubgraphInfo() = default;
    };

    using SubviewConsumersMap = DenseMap<mlir::Value, SubviewSubgraphInfo>;

    explicit OptimizeSubviewCopiesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    bool isOptimizableSubview(VPUIP::SubViewOp subview, const Byte& cmxSize) const;
    mlir::FailureOr<Byte> areSubviewConsumersCompatible(VPUIP::SubViewOp subview, const Byte& cmxSize) const;
    bool isParentCopyOptimizable(VPUIP::CopyOp copyParent, const SubviewSubgraphInfo& consumersInfo,
                                 const Byte& cmxSize) const;
    NDTypeInterface getNewCMXType(mlir::Value input, mlir::Type copyType);
    void moveSubviewsAfterCopy(mlir::Value input, SmallVector<VPUIP::SubViewOp>& subviews);
    mlir::Operation* createNewCopyOp(mlir::OpBuilder& builder, mlir::Value input, mlir::Type newOutputType,
                                     mlir::Location copyLoc);
};

// Must be 1xCx1x1 -> Subview -> 1xcx1x1 case and it must fit in CMX
bool OptimizeSubviewCopiesPass::isOptimizableSubview(VPUIP::SubViewOp subview, const Byte& cmxSize) const {
    auto subviewInput = subview->getOperand(0).getType().cast<NDTypeInterface>();
    auto subviewOutput = subview->getResult(0).getType().cast<NDTypeInterface>();
    const auto& nestLog = _log.nest();

    if (subviewInput.getMemoryKind() == VPU::MemoryKind::CMX_NN) {
        nestLog.trace("Subview's already in CMX");
        return false;
    }

    if (subviewInput.getRank() != 4 || subviewOutput.getRank() != 4) {
        nestLog.trace("Subview's input/output is not 4D");
        return false;
    }

    const auto inShape = subviewInput.getShape();
    if (inShape[Dims4D::Act::H] != 1 || inShape[Dims4D::Act::W] != 1 || inShape[Dims4D::Act::N] != 1) {
        nestLog.trace("Subview's input must have data only on channel dimension");
        return false;
    }

    if (inShape[Dims4D::Act::C] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        nestLog.trace("Channel size of Subview's input {0} is larger than VPU_DIMENSION_LIMIT {1}.",
                      inShape[Dims4D::Act::C], VPU::NCEInvariant::VPU_DIMENSION_LIMIT);
        return false;
    }

    if (inShape[Dims4D::Act::C] == subviewOutput.getShape()[Dims4D::Act::C]) {
        nestLog.trace("Subview is not slicing the channel dimension");
        return false;
    }

    if (VPU::getRequiredCMXSize({subviewInput}) > cmxSize) {
        nestLog.trace("Subview input does not fit in CMX");
        return false;
    }

    // TODO: Can be removed when RMS clustering is supported E#-142502
    auto parentCopy = subview->getOperand(0).getDefiningOp<VPUIP::CopyOp>();
    if (parentCopy != nullptr) {
        auto potentialSWOp = parentCopy->getOperand(0).getDefiningOp();
        while (mlir::isa_and_nonnull<VPUIP::PermuteCastOp, VPUIP::GenericReshapeOp>(potentialSWOp)) {
            potentialSWOp = potentialSWOp->getOperand(0).getDefiningOp();
        }
        if (auto swKernelOp = mlir::dyn_cast_or_null<VPUIP::SwKernelOp>(potentialSWOp)) {
            auto kernelFunction = swKernelOp.getKernelFunction();
            if (kernelFunction.getLeafReference().str() == "builtin_RMS") {
                nestLog.trace(
                        "RMS is not applicable for subview copy optimization because clustering is not supported.");
                return false;
            }
        }
    }

    return true;
}

mlir::FailureOr<Byte> OptimizeSubviewCopiesPass::areSubviewConsumersCompatible(VPUIP::SubViewOp subview,
                                                                               const Byte& cmxSize) const {
    auto subviewOutput = subview->getResult(0);
    bool hasDistributedConsumers = false;
    bool hasNonDistributedConsumers = false;

    Byte maxRequiredCMX = Byte(0);

    const auto& nestLog = _log.nest();

    for (const auto& child : subviewOutput.getUsers()) {
        if (!mlir::isa_and_nonnull<VPUIP::CopyOp>(child)) {
            nestLog.trace("Subview has non-CopyOp child.");
            return mlir::failure();
        }

        auto copyOut = child->getResult(0);
        auto copyOutType = copyOut.getType().cast<NDTypeInterface>();

        if (copyOutType.getMemoryKind() != VPU::MemoryKind::CMX_NN) {
            nestLog.trace("Copy op is not to NNCMX.");
            return mlir::failure();
        }

        auto distributedCopyOut = copyOutType.dyn_cast<VPU::DistributedTypeInterface>();
        if (distributedCopyOut != nullptr && distributedCopyOut.containsDistributedTypes()) {
            auto distribution = distributedCopyOut.getDistributedTypes()
                                        .front()
                                        .cast<VPUIP::DistributedBufferType>()
                                        .getDistribution();
            if (distribution.getMode().getValue() != VPU::DistributionMode::DUPLICATED) {
                nestLog.trace("Distributed Copy op is not DUPLICATED.");
                return mlir::failure();
            }

            hasDistributedConsumers = true;
        } else {
            hasNonDistributedConsumers = true;
        }

        for (const auto& grandchild : copyOut.getUsers()) {
            auto convOp = mlir::dyn_cast_or_null<VPUIP::NCEClusterTaskOp>(grandchild);
            if (convOp == nullptr || convOp.getTaskType() != VPUIP::NCETaskType::CONV) {
                nestLog.trace("Copy consumer is not a Conv NCEClusterTask.");
                return mlir::failure();
            }

            // skip the case of topbuf -> Slice -> Copy -> NCE(Weights/WT)
            // Weights/WT need to be changed together
            if (convOp->getOperand(0).getDefiningOp() != child) {
                nestLog.trace("Copy op output is not input activation for Conv NCEClusterTask.");
                return mlir::failure();
            }

            auto requiredCMXSize = VPUIP::getRequiredCMXSize(convOp);
            requiredCMXSize -= getTotalSize(child->getResult(0));
            requiredCMXSize += getTotalSize(subview.getSource());

            if (requiredCMXSize > cmxSize) {
                nestLog.trace("NCE Op does not fit in CMX with subview top buffer");
                return mlir::failure();
            }

            if (requiredCMXSize > maxRequiredCMX) {
                maxRequiredCMX = requiredCMXSize;
            }
        }
    }

    if (hasDistributedConsumers == hasNonDistributedConsumers) {
        nestLog.trace("Subview has a mix of distributed and non-distributed consumers");
        return mlir::failure();
    }

    return maxRequiredCMX;
}

mlir::Operation* OptimizeSubviewCopiesPass::createNewCopyOp(mlir::OpBuilder& builder, mlir::Value input,
                                                            mlir::Type newOutputType, mlir::Location copyLoc) {
    auto distributedCopy = newOutputType.dyn_cast<VPU::DistributedTypeInterface>();
    if (distributedCopy == nullptr || !distributedCopy.containsDistributedTypes()) {
        auto newCMXBuff = builder.create<mlir::memref::AllocOp>(appendLoc(copyLoc, "_top_buf"),
                                                                newOutputType.cast<mlir::MemRefType>());
        return builder.create<VPUIP::CopyOp>(copyLoc, newOutputType, input, newCMXBuff);
    }

    auto newCMXBuff =
            builder.create<VPURT::AllocDistributed>(appendLoc(copyLoc, "_top_buf"), newOutputType, nullptr, nullptr);

    return builder.create<VPUIP::CopyOp>(copyLoc, newOutputType, input, newCMXBuff);
}

NDTypeInterface OptimizeSubviewCopiesPass::getNewCMXType(mlir::Value input, mlir::Type copyType) {
    auto inputShape = input.getType().cast<NDTypeInterface>().getShape();

    auto prevCopyOutType = copyType.cast<NDTypeInterface>();
    auto distributedCopy = copyType.dyn_cast<VPU::DistributedTypeInterface>();
    if (distributedCopy == nullptr || !distributedCopy.containsDistributedTypes()) {
        return prevCopyOutType.changeShape(inputShape);
    }

    auto outputType = distributedCopy.getDistributedTypes().front().cast<VPUIP::DistributedBufferType>();
    const auto oldDistribution = outputType.getDistribution();
    if (!VPU::isDistributedAttrWithExplicitShapesAndOffsets(outputType.getDistribution())) {
        return outputType.changeShape(inputShape);
    }

    const auto newDistribution = VPU::getNonOverlappedDistributedAttr(
            inputShape, oldDistribution.getMode(), oldDistribution.getNumTiles(), oldDistribution.getNumClusters(),
            oldDistribution.getAlignment(), oldDistribution.getUniformDistributedSegments(),
            prevCopyOutType.getContext());
    return outputType.changeShapeForExplicitDistribution(inputShape, newDistribution);
}

void OptimizeSubviewCopiesPass::moveSubviewsAfterCopy(mlir::Value input, SmallVector<VPUIP::SubViewOp>& subviews) {
    mlir::OpBuilder builder(subviews[0]);
    const auto& nestLog = _log.nest();

    auto copyType = mlir::cast<VPUIP::CopyOp>(*subviews[0]->getUsers().begin())->getResult(0).getType();
    auto newType = getNewCMXType(input, copyType);

    auto subviewInput = input;
    auto inputType = input.getType().cast<NDTypeInterface>();
    if (inputType.getMemoryKind() != VPU::MemoryKind::CMX_NN) {
        nestLog.trace("Creating new CopyOp.");

        auto newCopyOp = createNewCopyOp(builder, input, newType, subviews[0]->getLoc());

        subviewInput = newCopyOp->getResult(0);
    } else {
        newType = newType.changeStrides(inputType.getStrides());
        auto distributedCopy = copyType.dyn_cast<VPU::DistributedTypeInterface>();
        if (distributedCopy != nullptr && distributedCopy.containsDistributedTypes()) {
            subviewInput = builder.createOrFold<VPUIP::DistributedCastOp>(subviews[0]->getLoc(), newType, input);
        }
    }

    nestLog.trace("Move Subviews to CMX");

    for (auto& subview : subviews) {
        nestLog.trace("Replace Copy outputs with Subview output {0}", subview->getLoc());

        auto cmxSubview = builder.create<VPUIP::SubViewOp>(appendLoc(subview->getLoc(), "_cmxsubview"), subviewInput,
                                                           subview.getStaticOffsetsAttr(), subview.getStaticSizesAttr(),
                                                           subview.getStaticStridesAttr());

        for (auto copyOp : llvm::make_early_inc_range(subview->getResult(0).getUsers())) {
            copyOp->getResult(0).replaceAllUsesWith(cmxSubview->getResult(0));
            copyOp->erase();
        }
    }

    _log.trace("Successfully moved Subview after all child copy ops.");
}

bool OptimizeSubviewCopiesPass::isParentCopyOptimizable(VPUIP::CopyOp copyParent,
                                                        const SubviewSubgraphInfo& consumersInfo,
                                                        const Byte& cmxSize) const {
    auto copyOutput = copyParent->getResult(0);
    const auto& subviews = consumersInfo.subviews;

    const auto& nestLog = _log.nest();

    const auto copyNumUsers =
            static_cast<size_t>(std::distance(copyOutput.getUsers().begin(), copyOutput.getUsers().end()));
    VPUX_THROW_WHEN(copyNumUsers < subviews.size(), "Copy op is not parent to all the subviews; malformed pattern.");
    if (copyNumUsers > subviews.size()) {
        nestLog.trace("Parent CopyOp has other consumers apart from the subviews, cannot optimize it out.");
        return false;
    }

    auto copyInput = copyParent->getOperand(0);
    auto copyInputType = copyInput.getType().cast<NDTypeInterface>();
    if (copyInputType.getMemoryKind() != VPU::MemoryKind::CMX_NN) {
        nestLog.trace("Parent Copy op is not from NNCMX.");
        return false;
    }

    auto distributedCopyIn = copyInputType.dyn_cast<VPU::DistributedTypeInterface>();
    if (distributedCopyIn != nullptr && distributedCopyIn.containsDistributedTypes()) {
        auto distribution =
                distributedCopyIn.getDistributedTypes().front().cast<VPUIP::DistributedBufferType>().getDistribution();
        if (!VPU::isDuplicated(distribution)) {
            nestLog.trace("Parent Copy op is not DUPLICATED. Cannot optimize it out.");
            return false;
        }
    }

    const auto copyInputSize = getTotalSize(copyInput);
    const auto copyOutputSize = getTotalSize(copyOutput);
    nestLog.trace("In size {0}, out size {1}", copyInputSize.count(), copyOutputSize.count());
    if (copyInputSize > copyOutputSize) {
        const auto requiredCMXSize = consumersInfo.maxRequiredCMX + copyInputSize - copyOutputSize;

        if (requiredCMXSize > cmxSize) {
            nestLog.trace("Cannot optimize out parent copy; will not fit in CMX");
            return false;
        }
    }

    return true;
}

//
// safeRunOnFunc
//

void OptimizeSubviewCopiesPass::safeRunOnFunc() {
    auto func = getOperation();

    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto cmxSize = VPU::getTotalCMXSize(module);

    const auto& nestLog = _log.nest();

    SubviewConsumersMap sourceToSubviewMap;

    func->walk([&](VPUIP::SubViewOp subview) {
        _log.trace("Got '{0}' at '{1}'", subview->getName(), subview->getLoc());
        if (!isOptimizableSubview(subview, cmxSize)) {
            nestLog.trace("Subview cannot be moved to CMX.");
            return;
        }

        const auto maxRequiredCMXRes = areSubviewConsumersCompatible(subview, cmxSize);
        if (mlir::failed(maxRequiredCMXRes)) {
            nestLog.trace("Subview consumers pattern is invalid.");
            return;
        }

        _log.trace("Valid pattern, can move subview to CMX.");

        auto subviewInput = subview->getOperand(0);
        if (sourceToSubviewMap.find(subviewInput) == sourceToSubviewMap.end()) {
            sourceToSubviewMap[subviewInput] =
                    SubviewSubgraphInfo(maxRequiredCMXRes.value(), SmallVector<VPUIP::SubViewOp>{subview});
        } else {
            auto& subviewSubgraph = sourceToSubviewMap[subviewInput];
            subviewSubgraph.subviews.push_back(subview);
            if (subviewSubgraph.maxRequiredCMX < maxRequiredCMXRes.value()) {
                subviewSubgraph.maxRequiredCMX = maxRequiredCMXRes.value();
            }
        }
    });

    for (auto subviewPattern : sourceToSubviewMap) {
        auto subviewInput = subviewPattern.first;
        auto subviewInfo = subviewPattern.second;

        _log.trace("Rewrite subgraph with parent: {0}.", subviewInput);
        auto producerCopy = subviewInput.getDefiningOp<VPUIP::CopyOp>();
        const bool canOptimizeProducerCopy =
                producerCopy != nullptr && isParentCopyOptimizable(producerCopy, subviewInfo, cmxSize);
        if (canOptimizeProducerCopy) {
            _log.nest().trace("Copy parent of Subview can also be optimized out.");
            subviewInput = producerCopy->getOperand(0);
        }

        moveSubviewsAfterCopy(subviewInput, subviewInfo.subviews);

        for (auto& subview : llvm::make_early_inc_range(subviewInfo.subviews)) {
            subview->erase();
        }

        if (canOptimizeProducerCopy) {
            producerCopy->erase();
        }
    }
}

}  // namespace

//
// createOptimizeSubviewCopiesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createOptimizeSubviewCopiesPass(Logger log) {
    return std::make_unique<OptimizeSubviewCopiesPass>(log);
}
