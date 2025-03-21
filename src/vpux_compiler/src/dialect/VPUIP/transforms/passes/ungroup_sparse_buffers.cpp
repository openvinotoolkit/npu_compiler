//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/IRMapping.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_UNGROUPSPARSEBUFFERS
#define GEN_PASS_DEF_UNGROUPSPARSEBUFFERS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

// Set the types of the inner arguments to correspond to the individual buffer types
// and reinfers the return types of the inner ops, now that they no longer receive
// sparse buffers
void adaptNCEClusterTilingBody(VPUIP::NCEClusterTilingOp tilingOp, ArrayRef<mlir::Value> individualOperands) {
    auto& body = tilingOp.getBody();
    for (auto p : body.getArguments() | indexed) {
        auto arg = p.value();
        const auto argIndex = p.index();
        if (arg.getType().isa<VPUIP::SparseBufferType>()) {
            auto operandType = individualOperands[argIndex].getType();
            if (auto distOperandType = operandType.dyn_cast<VPUIP::DistributedBufferType>()) {
                operandType = distOperandType.getCompactType();
            }
            arg.setType(operandType);
        }
    }

    auto& block = body.front();
    for (auto& op : block.getOperations()) {
        vpux::inferReturnTypes(&op, vpux::InferShapedTypeMode::ALL);
    }
}

mlir::Operation* createUngroupedOp(Logger& log, mlir::OpBuilder& builder, mlir::Operation* op,
                                   ArrayRef<mlir::Value> sparseOperands, ArrayRef<mlir::Value> individualOperands,
                                   ArrayRef<mlir::Type> individualResultTypes) {
    if (individualOperands.empty() && individualResultTypes.empty()) {
        return nullptr;
    }

    log.nest().trace("Creating ungrouped op {0} with {1} operands and {2} result types", op->getName(),
                     individualOperands.size(), individualResultTypes.size());

    mlir::IRMapping mapper;
    mapper.map(sparseOperands, individualOperands);
    auto individualOp = builder.clone(*op, mapper);

    if (auto tilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(individualOp)) {
        adaptNCEClusterTilingBody(tilingOp, individualOperands);
    }

    if (!individualResultTypes.empty()) {
        int64_t sparseResultIdx = 0;
        for (auto result : individualOp->getResults()) {
            if (!result.getType().isa<VPUIP::SparseBufferType>()) {
                continue;
            }
            result.setType(individualResultTypes[sparseResultIdx++]);
        }
    }
    return individualOp;
}

void ungroupOperation(Logger& log, mlir::OpBuilder& builder, mlir::Operation* op, ArrayRef<mlir::Value> sparseOperands,
                      ArrayRef<mlir::Value> sparseResults) {
    VPUX_THROW_UNLESS(sparseResults.size() == 1, "Expected only one sparse result, got {0}", sparseResults.size());

    SmallVector<mlir::Value> dataOperands;
    SmallVector<mlir::Value> smOperands;
    SmallVector<mlir::Value> seTableOperands;
    for (auto operand : sparseOperands) {
        auto ungroupOp = builder.create<VPUIP::UngroupSparseBufferOp>(op->getLoc(), operand);
        dataOperands.push_back(ungroupOp.getData());
        if (ungroupOp.getSparsityMap() != nullptr) {
            smOperands.push_back(ungroupOp.getSparsityMap());
        }
        if (ungroupOp.getStorageElementTable() != nullptr) {
            seTableOperands.push_back(ungroupOp.getStorageElementTable());
        }
    }

    SmallVector<mlir::Type> dataResultTypes;
    SmallVector<mlir::Type> smResultTypes;
    SmallVector<mlir::Type> seTableResultTypes;
    SmallVector<mlir::UnitAttr> isWeights;
    SmallVector<VPUIP::SparsityCompressionAttr> sparsityCompressions;
    SmallVector<VPU::SEAttr> seAttrs;
    for (auto result : sparseResults) {
        auto sparseType = result.getType().cast<VPUIP::SparseBufferType>();
        dataResultTypes.push_back(sparseType.getData());
        if (sparseType.getSparsityMap() != nullptr) {
            smResultTypes.push_back(sparseType.getSparsityMap());
        }
        if (sparseType.getStorageElementTable() != nullptr) {
            seTableResultTypes.push_back(sparseType.getStorageElementTable());
        }
        isWeights.push_back(sparseType.getIsWeights());
        sparsityCompressions.push_back(sparseType.getSparsityCompression());
        seAttrs.push_back(sparseType.getSeAttr());
    }

    auto dataOp = createUngroupedOp(log, builder, op, sparseOperands, dataOperands, dataResultTypes);
    auto smOp = createUngroupedOp(log, builder, op, sparseOperands, smOperands, smResultTypes);
    auto seTableOp = createUngroupedOp(log, builder, op, sparseOperands, seTableOperands, seTableResultTypes);

    auto dataResult = dataOp->getResult(0);
    auto smResult = (smOp != nullptr) ? smOp->getResult(0) : nullptr;
    auto seTableResult = (seTableOp != nullptr) ? seTableOp->getResult(0) : nullptr;
    auto isWeightsAttr = (isWeights.size() > 0) ? isWeights[0] : nullptr;
    auto sparsityCompressionAttr = (sparsityCompressions.size() > 0) ? sparsityCompressions[0] : nullptr;
    auto seAttr = (seAttrs.size() > 0) ? seAttrs[0] : nullptr;

    auto groupOp = builder.create<VPUIP::GroupSparseBufferOp>(op->getLoc(), dataResult, smResult, seTableResult,
                                                              isWeightsAttr, sparsityCompressionAttr, seAttr);
    op->getResult(0).replaceAllUsesExcept(groupOp.getOutput(), llvm::SmallPtrSet<mlir::Operation*, 1>{groupOp});
    op->erase();
}

//
// RemoveGroupUngroup
//

class RemoveGroupUngroupRewriter final : public mlir::OpRewritePattern<VPUIP::GroupSparseBufferOp> {
public:
    RemoveGroupUngroupRewriter(mlir::MLIRContext* ctx): mlir::OpRewritePattern<VPUIP::GroupSparseBufferOp>(ctx) {
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::GroupSparseBufferOp groupOp,
                                        mlir::PatternRewriter& /*rewriter*/) const final {
        if (llvm::any_of(groupOp.getOutput().getUsers(), [](mlir::Operation* userOp) {
                return !mlir::isa<VPUIP::UngroupSparseBufferOp>(userOp);
            })) {
            return mlir::failure();
        }

        const auto operands = groupOp.getOperands();
        for (auto userOp : groupOp.getOutput().getUsers()) {
            for (auto userResult : userOp->getResults() | indexed) {
                userResult.value().replaceAllUsesWith(operands[userResult.index()]);
            }
        }

        return mlir::success();
    }
};

//
// UngroupSparseBuffers
//

class UngroupSparseBuffers final : public VPUIP::impl::UngroupSparseBuffersBase<UngroupSparseBuffers> {
public:
    explicit UngroupSparseBuffers(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UngroupSparseBuffers::safeRunOnFunc() {
    auto func = getOperation();

    const auto getSparseValues = [](mlir::ValueRange values) -> SmallVector<mlir::Value> {
        SmallVector<mlir::Value> sparseValues;
        for (auto value : values) {
            if (value.getType().isa<VPUIP::SparseBufferType>()) {
                sparseValues.push_back(value);
            }
        }
        return sparseValues;
    };

    // Insert group / ungroup operations around each instance of sparse types
    for (auto& op : llvm::make_early_inc_range(func.getOps())) {
        if (mlir::isa<VPUIP::GroupSparseBufferOp, VPUIP::UngroupSparseBufferOp>(op)) {
            continue;
        }

        // Operations that are found inside NCEClusterTiling will be treated separately,
        // when treating the parent operation
        if (op.getParentOfType<VPUIP::NCEClusterTilingOp>() != nullptr) {
            continue;
        }

        const auto sparseOperands = getSparseValues(op.getOperands());
        const auto sparseResults = getSparseValues(op.getResults());
        if (sparseOperands.empty() && sparseResults.empty()) {
            continue;
        }

        _log.trace("Ungrouping operation '{0}' at '{1}'", op.getName(), op.getLoc());

        mlir::OpBuilder builder(&op);
        ungroupOperation(_log, builder, &op, sparseOperands, sparseResults);
    }

    // Remove pairs of group - ungroup operations
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<RemoveGroupUngroupRewriter>(&ctx);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createUngroupSparseBuffersPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createUngroupSparseBuffersPass(Logger log) {
    return std::make_unique<UngroupSparseBuffers>(log);
}
