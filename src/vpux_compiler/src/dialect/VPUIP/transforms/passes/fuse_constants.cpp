//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/constant_fusion.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_FUSECONSTANTS
#define GEN_PASS_DEF_FUSECONSTANTS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;
namespace {

//
// FuseConstants
//

class FuseConstants final : public mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp> {
public:
    FuseConstants(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::NCEClusterTaskOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::NCEClusterTaskOp nceOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::RankedTensorType getFusedConstantType(vpux::ConstantFusing::ConstantVector& constantVector,
                                                mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

mlir::Value createAllocOp(Const::DeclareOp declOp, VPURT::AllocDistributed allocDistributed,
                          mlir::PatternRewriter& rewriter) {
    if (allocDistributed) {
        auto origType = mlir::cast<vpux::VPUIP::DistributedBufferType>(allocDistributed.getType());
        auto newType = vpux::ConstantFusing::getDistributedBufferType(origType, declOp, rewriter);
        auto distributedBufferType = mlir::cast<vpux::VPUIP::DistributedBufferType>(newType);
        return rewriter.create<VPURT::AllocDistributed>(declOp.getLoc(), distributedBufferType, nullptr, nullptr)
                .getBuffer();

    } else {
        const auto type = mlir::cast<vpux::NDTypeInterface>(declOp.getOutput().getType());
        vpux::IndexedSymbolAttr memKindAttr =
                IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), 0);
        auto newType = type.changeMemSpace(memKindAttr);
        auto memrefType = mlir::cast<mlir::MemRefType>(newType);
        return rewriter.create<mlir::memref::AllocOp>(declOp.getLoc(), memrefType).getMemref();
    }
}

VPUIP::CopyOp createFusedCopyOp(mlir::Value allocDefiningOp, Const::DeclareOp declOp, mlir::PatternRewriter& rewriter) {
    VPUIP::CopyOp fusedCopyOp = nullptr;
    if (auto allocOp = allocDefiningOp.getDefiningOp<VPURT::AllocDistributed>()) {
        fusedCopyOp = rewriter.create<VPUIP::CopyOp>(appendLoc(declOp.getLoc(), "_fused_tile"), declOp.getResult(),
                                                     allocOp.getBuffer());
    } else if (auto allocOp = allocDefiningOp.getDefiningOp<mlir::memref::AllocOp>()) {
        fusedCopyOp = rewriter.create<VPUIP::CopyOp>(declOp->getLoc(), declOp.getOutput(), allocOp.getMemref());
    } else {
        VPUX_THROW("Unrecognized allocDefiningOp encountered");
    }
    return fusedCopyOp;
}

void replaceConstantsWithFusedConstant(vpux::ConstantFusing::ConstantVector& constantVector,
                                       mlir::PatternRewriter& rewriter, VPUIP::CopyOp newCopyOp) {
    auto opElementType = mlir::cast<vpux::NDTypeInterface>(newCopyOp.getOutput().getType()).getElementType();
    const auto opElementSizeBytes = opElementType.getIntOrFloatBitWidth() / CHAR_BIT;

    // 5.  Replace constants constant with sequence fused_constant -> subview -> view
    int64_t offset = 0;
    vpux::Byte size(0);
    for (size_t i = 0; i < constantVector.size(); ++i) {
        auto constant = constantVector[i].second;
        if (constant == nullptr) {
            continue;
        }
        size = vpux::getTotalSize(constant->getOpResult(0)) / opElementSizeBytes;
        SmallVector<int64_t> subtensor({1, 1, 1, size.count()});
        auto offsets = SmallVector<int64_t>{0, 0, 0, offset};
        auto copyOp = constantVector[i].first;
        auto subViewOp =
                rewriter.create<VPUIP::SubViewOp>(constant.getLoc(), newCopyOp.getOutput(), offsets, subtensor);
        rewriter.replaceOpWithNewOp<VPUIP::ViewOp>(copyOp, copyOp.getOutputBuff().getType(), subViewOp.getResult());
        offset += size.count();
    }
}

// For a given layer type we need to determine the constant fusing order given the presence (or not) of weights,
// weights sparsity map, weight table. In certain cases it might not be possible to fuse
// the constants e.g for case when layer weights are not constants and are in graphfile or if the declare or copyop
// couldn't be found in such case matchFailed is returned with the error message
// E#45170 - Update the logic to make constant selection generic
mlir::LogicalResult getInputsInFusingOrder(VPUIP::NCEClusterTaskOp& nceOp,
                                           vpux::ConstantFusing::ConstantVector& constantVector,
                                           SmallVector<VPURT::AllocDistributed>& tilingVector,
                                           mlir::PatternRewriter& rewriter) {
    VPUIP::CopyOp copyOp = nullptr;
    Const::DeclareOp declareOp = nullptr;

    VPURT::AllocDistributed allocDistributed = nullptr;

    auto resetTemporaries = [&]() {
        copyOp = nullptr;
        declareOp = nullptr;
        allocDistributed = nullptr;
    };

    vpux::ConstantFusing::getCopyAndDeclareOpForFusion(nceOp.getWeightTable(), copyOp, declareOp, allocDistributed);
    if (copyOp != nullptr && declareOp != nullptr) {
        constantVector[0] = {copyOp, declareOp};
        tilingVector[0] = allocDistributed;
    } else {
        return matchFailed(rewriter, nceOp, "Couldn't find weight table");
    }

    if (nceOp.getWeights() != nullptr) {
        resetTemporaries();

        vpux::ConstantFusing::getCopyAndDeclareOpForFusion(nceOp.getWeights(), copyOp, declareOp, allocDistributed);
        if (copyOp == nullptr) {
            return matchFailed(rewriter, nceOp, "Weights Copy Op missing");
        }
        // TODO Handling shared weights with weight table. Ticket E#61458
        if (!copyOp->hasOneUse()) {
            return matchFailed(rewriter, nceOp, "Weights Copy Op has more than one use");
        }

        if (declareOp != nullptr) {
            constantVector[1] = {copyOp, declareOp};
            tilingVector[1] = allocDistributed;
        } else {
            // Special condition when weights come in from a different source
            // e.g. Activation tensor
            return matchFailed(rewriter, nceOp, "Non constant layer weights");
        }
    }

    if (nceOp.getWeightsSparsityMap() != nullptr) {
        resetTemporaries();
        vpux::ConstantFusing::getCopyAndDeclareOpForFusion(nceOp.getWeightsSparsityMap(), copyOp, declareOp,
                                                           allocDistributed);
        if (copyOp == nullptr) {
            return matchFailed(rewriter, nceOp, "Weights sparsity map Copy Op missing");
        }

        if (!copyOp->hasOneUse()) {
            return matchFailed(rewriter, nceOp, "Weights sparsity copy op has more than one use");
        }

        if (declareOp != nullptr) {
            constantVector[2] = {copyOp, declareOp};
            tilingVector[2] = allocDistributed;
        } else {
            return matchFailed(rewriter, nceOp, "The layer weights sparsity map is not constant");
        }
    }
    return mlir::success();
}

mlir::RankedTensorType FuseConstants::getFusedConstantType(vpux::ConstantFusing::ConstantVector& constantVector,
                                                           mlir::PatternRewriter& rewriter) const {
    int64_t totalTensorSize = 0;
    unsigned jointF16ConstantSize = 0;
    for (auto& constant : constantVector) {
        if (constant.second != nullptr) {
            auto contentType = mlir::cast<vpux::NDTypeInterface>(constant.second.getType());
            auto elemType = contentType.getElementType();
            if (elemType.isF16()) {
                jointF16ConstantSize += vpux::getTotalSize(constant.second->getOpResult(0)).count();
            }
            totalTensorSize += vpux::getTotalSize(constant.second->getOpResult(0)).count();
        }
    }

    mlir::RankedTensorType fusedTensorType = nullptr;
    // use F16 type if the fused constant
    // is to be dominated by this type
    if (jointF16ConstantSize * 2 > totalTensorSize) {
        auto fusedConstantElementType = mlir::Float16Type::get(rewriter.getContext());
        SmallVector<int64_t> fusedConstShape({1, 1, 1, totalTensorSize / 2});
        fusedTensorType = mlir::RankedTensorType::get(fusedConstShape, fusedConstantElementType);
    } else {
        auto fusedConstantElementType = getUInt8Type(rewriter.getContext());
        SmallVector<int64_t> fusedConstShape({1, 1, 1, totalTensorSize});
        fusedTensorType = mlir::RankedTensorType::get(fusedConstShape, fusedConstantElementType);
    }

    return fusedTensorType;
}

mlir::LogicalResult FuseConstants::matchAndRewrite(VPUIP::NCEClusterTaskOp nceOp,
                                                   mlir::PatternRewriter& rewriter) const {
    if (nceOp.getWeightTable() == nullptr) {
        return mlir::failure();
    }

    if (nceOp->hasAttr(vpux::ConstantFusing::constantsFused)) {
        return mlir::failure();
    }

    if (nceOp.getTaskType() == VPUIP::NCETaskType::ELTWISE || nceOp.getTaskType() == VPUIP::NCETaskType::AVEPOOL) {
        return mlir::failure();
    }
    // TODO Adapt pass in order to support zero offset weight table cases. Ticket E#157078
    if (nceOp.getIsZeroOffsetWeightsTable()) {
        _log.trace("Constant fusion is not yet supported for weights table reuse.");
        return mlir::failure();
    }

    // 1. Find constant inputs
    vpux::ConstantFusing::ConstantVector constantVector(vpux::ConstantFusing::numberOfConstantsToFuse,
                                                        {nullptr, nullptr});
    SmallVector<VPURT::AllocDistributed> tilingVector(vpux::ConstantFusing::numberOfConstantsToFuse, nullptr);
    if (getInputsInFusingOrder(nceOp, constantVector, tilingVector, rewriter).failed()) {
        constantVector.clear();
        tilingVector.clear();
        return mlir::failure();
    }
    // for ContentAttr with Fuse transform base content doesn't actually matter,
    // we simply have to pass something to satisfy API requirements. Since weightTable is
    // assumed to be always present we use its base content.
    VPUX_THROW_UNLESS(constantVector[0].second != nullptr, "No weight table for fusing");
    const auto fakeBaseContent = constantVector[0].second.getContentAttr().getBaseContent();
    const auto newLoc = appendLoc(nceOp.getLoc(), "_fused_constant");
    auto newContentAttr = Const::ContentAttr::get(fakeBaseContent);
    auto tensorType = getFusedConstantType(constantVector, rewriter);
    auto const weightsTable =
            constantVector[0].second != nullptr ? constantVector[0].second.getContentAttr() : Const::ContentAttr{};
    auto const weights =
            constantVector[1].second != nullptr ? constantVector[1].second.getContentAttr() : Const::ContentAttr{};
    auto const sparsity =
            constantVector[2].second != nullptr ? constantVector[2].second.getContentAttr() : Const::ContentAttr{};
    auto fusedContentAttr = newContentAttr.transform().fuse(tensorType, weightsTable, weights, sparsity, {}).get();
    // 2. Create fused constant of u8 type with size of weights + weights sparsity map + weights table + activation
    // window Fill it with the original binary data
    VPUX_THROW_UNLESS(tensorType != nullptr, "Couldn't fuse constant tensor type");

    // 3. Build new constant memref
    auto fusedTensorTypeMemref = vpux::convertToMemRef(tensorType);
    auto fusedConstant = rewriter.create<Const::DeclareOp>(newLoc, fusedTensorTypeMemref, std::move(fusedContentAttr));

    // 4. build new AllocOp
    auto allocOp = createAllocOp(fusedConstant, tilingVector[0], rewriter);

    // 5. create CopyOp, copy constant to allocated buffer
    auto copyOp = createFusedCopyOp(allocOp, fusedConstant, rewriter);

    // 6.  Replace constants with sequence fused_constant -> subview -> viewOp
    replaceConstantsWithFusedConstant(constantVector, rewriter, copyOp);

    // 7. Set constantsFused attribute so we can check the fusion status (fused or unfused) of the current layer
    // in patch_weight_table pass in VPUIP Dialect
    nceOp->setAttr(vpux::ConstantFusing::constantsFused, mlir::BoolAttr::get(nceOp.getContext(), true));
    return mlir::success();
}

//
// FuseConstantsPass
//

class FuseConstantsPass final : public VPUIP::impl::FuseConstantsBase<FuseConstantsPass> {
public:
    explicit FuseConstantsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void FuseConstantsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<FuseConstants>(&ctx, _log);

    auto func = getOperation();
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createFuseConstantsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createFuseConstantsPass(Logger log) {
    return std::make_unique<FuseConstantsPass>(log);
}
