//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/SetOperations.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_OPTIMIZESHAREDINPUTCOPYFORCONCAT
#define GEN_PASS_DEF_OPTIMIZESHAREDINPUTCOPYFORCONCAT
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

struct UserProcessingData {
    VPU::SliceOp slice;
    mlir::Operation* copyOp;
    SmallVector<mlir::Value> newInputs;
    SmallVector<SmallVector<int64_t>> newConcatOffsets;
    SmallVector<mlir::Operation*> newDefOps;
};

bool checkMemoryKind(mlir::Value value, VPU::MemoryKind kind) {
    return mlir::cast<vpux::NDTypeInterface>(value.getType()).getMemoryKind() == kind;
}

bool isCopyCMX2DDR(mlir::Operation* op) {
    if (!mlir::isa_and_nonnull<VPU::CopyOp>(op)) {
        return false;
    }
    return checkMemoryKind(op->getOperand(0), VPU::MemoryKind::CMX_NN) &&
           checkMemoryKind(op->getResult(0), VPU::MemoryKind::DDR);
}

bool isCopyDDR2CMX(mlir::Operation* op) {
    if (!mlir::isa_and_nonnull<VPU::CopyOp>(op)) {
        return false;
    }
    return checkMemoryKind(op->getOperand(0), VPU::MemoryKind::DDR) &&
           checkMemoryKind(op->getResult(0), VPU::MemoryKind::CMX_NN);
}

SmallVector<int64_t> getOffsetsFromConcat(int64_t inputIdx, VPU::ConcatOp concatOp) {
    auto offsets = parseIntArrayOfArrayAttr<int64_t>(concatOp.getStaticOffsets().value());
    VPUX_THROW_WHEN(checked_cast<int64_t>(offsets.size()) < inputIdx, "Invalid input index {0} for Concat op at '{1}'",
                    inputIdx, concatOp->getLoc());
    return offsets[inputIdx];
}

NDTypeInterface getConcatDistributedType(VPU::DistributedTypeInterface origType, ShapeRef shape) {
    auto distributedDataType = mlir::cast<vpux::VPU::DistributedTensorType>(origType.getDistributedTypes().front());
    const auto typeComponents = TypeComponents().setShape(shape).setElementType(distributedDataType.getElementType());

    if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distributedDataType.getDistribution())) {
        auto distribution = distributedDataType.getDistribution();
        if (auto sparseType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(origType)) {
            distribution = VPU::getExplicitDistrAttrForActualDataFromSparseType(sparseType);
        }

        auto newDistributedAttr =
                getConcatExplicitDistributedAttrForNewShape(distribution, shape, origType.getContext());
        return mlir::cast<vpux::NDTypeInterface>(
                origType.changeTypeComponentsForExplicitDistribution(typeComponents, newDistributedAttr));
    }

    return mlir::cast<vpux::NDTypeInterface>(origType).changeTypeComponents(typeComponents);
}

//
// SharedCopyInputRewriter
//

class SharedCopyInputRewriter final : public mlir::OpRewritePattern<VPU::ConcatOp> {
public:
    SharedCopyInputRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::ConcatOp>(ctx), _log(log) {
        this->setDebugName("SharedCopyInputRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;
    bool meetConcatPattern(VPU::ConcatOp) const;

    std::optional<mlir::Value> createNewBranchInput(mlir::Value concatInput, int64_t concatInputIdx,
                                                    VPU::ConcatOp concat, VPU::SliceOp sliceUser,
                                                    SmallVector<int64_t>& newConcatOffset,
                                                    mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

mlir::LogicalResult SharedCopyInputRewriter::matchAndRewrite(VPU::ConcatOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    auto hasCopyInput = llvm::any_of(origOp.getInputs(), [&](const auto& input) {
        return isCopyCMX2DDR(input.getDefiningOp());
    });
    if (!hasCopyInput) {
        return mlir::failure();
    }

    SmallVector<VPU::ConcatOp> concatsWithSharedCopyInput;
    for (auto user : origOp->getUsers()) {
        if (auto concat = mlir::dyn_cast<VPU::ConcatOp>(user)) {
            if (meetConcatPattern(concat)) {
                concatsWithSharedCopyInput.push_back(concat);
            }
        }
    }

    if (concatsWithSharedCopyInput.size() < 2) {
        return mlir::failure();
    }

    auto ctx = origOp->getContext();
    _log.trace("process shared input copy at'{0}'", origOp->getLoc());

    /*
       Convert this subgraph:
         Copy(CMX2DDR)                       Copy(CMX2DDR)
              |                                |
            Concat                           Concat
           /     \                          /     \
      Concat0    Concat1                  Slice  Slice
         |         |             =>        |       |
       Slice     Slice                    Copy    Copy
         |         |                       |       |
       Copy(DDR2CMX)  Copy(DDR2CMX)     CMXConcat CMXConcat

    */

    for (auto concat : concatsWithSharedCopyInput) {
        _log.trace("propagate ops through concat '{0}'", concat.getLoc());
        llvm::DenseMap<mlir::Operation*, VPU::ConcatOp> newUserMapping;
        mlir::Operation* insertPoint = concat.getOperation();

        SmallVector<UserProcessingData> userDataList;
        SmallVector<mlir::Operation*> users(concat->getUsers().begin(), concat->getUsers().end());

        userDataList.reserve(users.size());

        for (auto user : users) {
            UserProcessingData userData;
            userData.slice = mlir::cast<VPU::SliceOp>(user);
            userData.copyOp = *user->getUsers().begin();

            for (auto item : concat.getInputs() | indexed) {
                auto input = item.value();
                auto inputIdx = item.index();
                SmallVector<int64_t> newConcatOffset;
                auto newInput =
                        createNewBranchInput(input, inputIdx, concat, userData.slice, newConcatOffset, rewriter);
                if (newInput.has_value()) {
                    userData.newInputs.push_back(newInput.value());
                    userData.newConcatOffsets.push_back(newConcatOffset);
                    userData.newDefOps.push_back(newInput.value().getDefiningOp());
                }
            }

            VPUX_THROW_WHEN(userData.newInputs.empty(), "new slice input is empty");
            userDataList.push_back(std::move(userData));
        }
        for (const auto& userData : userDataList) {
            for (auto* newDefOp : userData.newDefOps) {
                if (insertPoint != newDefOp) {
                    bool isBeforeResult = insertPoint->isBeforeInBlock(newDefOp);
                    if (isBeforeResult) {
                        insertPoint = newDefOp;
                    }
                }
            }
        }

        rewriter.setInsertionPointAfter(insertPoint);
        for (const auto& userData : userDataList) {
            auto newConcat = rewriter.create<VPU::ConcatOp>(userData.copyOp->getLoc(),
                                                            userData.copyOp->getResult(0).getType(), userData.newInputs,
                                                            getIntArrayOfArray(ctx, userData.newConcatOffsets));
            newUserMapping.insert({userData.copyOp, newConcat});
        }

        // Replace copy user with new concat op
        for (auto& item : newUserMapping) {
            rewriter.replaceOp(item.first, item.second);
        }
        // Remove concat op and its user
        for (auto user : llvm::make_early_inc_range(concat->getUsers())) {
            rewriter.eraseOp(user);
        }
        rewriter.eraseOp(concat);
    }
    return mlir::success();
}

bool SharedCopyInputRewriter::meetConcatPattern(VPU::ConcatOp concatOp) const {
    if (!concatOp.getStaticOffsets().has_value()) {
        // Only handle concat op with static offset
        return false;
    }
    auto concatAxes = vpux::VPU::getConcatAxes(concatOp);
    if (concatAxes.size() != 1) {
        // For concat on multi dims, the input and output tensor of the new copy has strides on multi dims,
        // which may cause accuracy regression when lowering to DMA op
        return false;
    }

    for (auto user : concatOp->getUsers()) {
        if (!user->hasOneUse()) {
            return false;
        }
        auto maybeSliceOp = mlir::dyn_cast<VPU::SliceOp>(user);
        if (maybeSliceOp == nullptr || !maybeSliceOp->hasOneUse()) {
            return false;
        }

        auto userOp = *maybeSliceOp->getUsers().begin();

        if (!isCopyDDR2CMX(userOp)) {
            return false;
        }

        auto outputType = mlir::cast<vpux::NDTypeInterface>(userOp->getResult(0).getType());
        if (vpux::VPU::NCEInvariant::hasDimensionExceedingVPULimit(outputType.getShape())) {
            _log.trace("Dimension sizes over VPU_DIMENSION_LIMIT are not supported for output shape {0}",
                       outputType.getShape());
            return false;
        }

        auto distributedOutput = mlir::dyn_cast_or_null<VPU::DistributedTensorType>(userOp->getResult(0).getType());
        if (distributedOutput != nullptr) {
            const auto distAttr = distributedOutput.getDistribution();
            const auto distMode = distAttr.getMode().getValue();
            if (distMode == VPU::DistributionMode::SEGMENTED || distMode == VPU::DistributionMode::OVERLAPPED) {
                const auto numTiles = distAttr.getNumTiles();
                const auto tilingScheme = parseIntArrayAttr<int64_t>(numTiles);
                auto tilingAxis = VPU::getDistributedTilingAxis(tilingScheme);
                if (concatAxes.contains(tilingAxis)) {
                    return false;
                }
            }
        }
    };

    return true;
}

std::optional<mlir::Value> SharedCopyInputRewriter::createNewBranchInput(mlir::Value concatInput,
                                                                         int64_t concatInputIdx, VPU::ConcatOp concat,
                                                                         VPU::SliceOp sliceUser,
                                                                         SmallVector<int64_t>& newConcatOffset,
                                                                         mlir::PatternRewriter& rewriter) const {
    const auto concatOffset = getOffsetsFromConcat(concatInputIdx, concat);
    const auto concatSize = to_small_vector(getShape(concatInput));

    const auto sliceOffset = parseIntArrayAttr<int64_t>(sliceUser.getStaticOffsets());
    const auto sliceSize = parseIntArrayAttr<int64_t>(sliceUser.getStaticSizes());

    // check the overlapp concat input has intersection with the slice output
    SmallVector<int64_t> overlappedOffset(concatSize.size(), 0);
    SmallVector<int64_t> overlappedSize(concatSize.size(), 0);

    for (auto i : irange(sliceOffset.size())) {
        overlappedOffset[i] = std::max(concatOffset[i], sliceOffset[i]);
        auto concatEndOffset = concatOffset[i] + concatSize[i];
        auto sliceEndOffset = sliceOffset[i] + sliceSize[i];
        auto overlappedEndOffset = std::min(sliceEndOffset, concatEndOffset);
        overlappedSize[i] = overlappedEndOffset - overlappedOffset[i];
    }
    auto hasLegalOverLapped = llvm::all_of(overlappedSize, [](const auto& size) {
        return size > 0;
    });

    if (!hasLegalOverLapped) {
        return std::nullopt;
    }
    // Convert overlapped offset to slice offset for the input
    SmallVector<int64_t> newSliceOffset(overlappedSize.size(), 0);
    SmallVector<int64_t> newSliceSize = std::move(overlappedSize);
    newConcatOffset.resize(concatOffset.size());
    for (auto i : irange(concatOffset.size())) {
        newSliceOffset[i] = overlappedOffset[i] - concatOffset[i];
        newConcatOffset[i] = std::max(static_cast<int64_t>(0), concatOffset[i] - sliceOffset[i]);
    }

    rewriter.setInsertionPointAfter(concat);
    auto newSlice = rewriter.create<VPU::SliceOp>(sliceUser.getLoc(), concatInput, newSliceOffset, newSliceSize);

    auto userCopyOp = *sliceUser->getUsers().begin();

    auto copyOutType = mlir::cast<vpux::NDTypeInterface>(userCopyOp->getResult(0).getType());

    if (vpux::VPU::NCEInvariant::hasDimensionExceedingVPULimit(copyOutType.getShape())) {
        _log.trace("Dimension sizes over VPU_DIMENSION_LIMIT are not supported for copy output shape {0}",
                   copyOutType.getShape());
        return std::nullopt;
    }

    if (!mlir::isa<VPU::DistributedTensorType>(copyOutType)) {
        return rewriter.create<VPU::CopyOp>(userCopyOp->getLoc(), newSlice, copyOutType.getMemSpace());
    }

    auto newShape = getShape(newSlice.getResult());
    auto newOutType = getConcatDistributedType(mlir::cast<vpux::VPU::DistributedTensorType>(copyOutType), newShape);
    auto newCopyOp =
            rewriter.create<VPU::CopyOp>(userCopyOp->getLoc(), newOutType, newSlice, copyOutType.getMemSpace());

    return newCopyOp->getResult(0);
}

//
// OptimizeSharedInputCopyForConcatPass
//

class OptimizeSharedInputCopyForConcatPass final :
        public VPU::impl::OptimizeSharedInputCopyForConcatBase<OptimizeSharedInputCopyForConcatPass> {
public:
    explicit OptimizeSharedInputCopyForConcatPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() override;

private:
};

void OptimizeSharedInputCopyForConcatPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SharedCopyInputRewriter>(&ctx, _log);

    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeSharedInputCopyForConcatPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createOptimizeSharedInputCopyForConcatPass(Logger log) {
    return std::make_unique<OptimizeSharedInputCopyForConcatPass>(log);
}
