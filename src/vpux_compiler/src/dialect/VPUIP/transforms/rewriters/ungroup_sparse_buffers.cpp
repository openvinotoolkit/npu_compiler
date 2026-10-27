//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/rewriters.hpp"
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

mlir::Operation* createUngroupedOp(Logger log, mlir::OpBuilder& builder, mlir::Operation* op,
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

    if (!individualResultTypes.empty()) {
        int64_t sparseResultIdx = 0;
        for (auto result : individualOp->getResults()) {
            if (!mlir::isa<vpux::VPUIP::SparseBufferType>(result.getType())) {
                continue;
            }
            result.setType(individualResultTypes[sparseResultIdx++]);
        }
    }
    return individualOp;
}

void ungroupOperation(Logger log, mlir::PatternRewriter& rewriter, mlir::Operation* op,
                      ArrayRef<mlir::Value> sparseOperands, ArrayRef<mlir::Value> sparseResults) {
    VPUX_THROW_UNLESS(sparseResults.size() == 1, "Expected only one sparse result, {0} got {1}", op->getName(),
                      sparseResults.size());

    SmallVector<mlir::Value> dataOperands;
    SmallVector<mlir::Value> smOperands;
    SmallVector<mlir::Value> seTableOperands;
    for (auto operand : sparseOperands) {
        auto ungroupOp = VPUIP::UngroupSparseBufferOp::create(rewriter, op->getLoc(), operand);
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
        auto sparseType = mlir::cast<vpux::VPUIP::SparseBufferType>(result.getType());
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

    auto dataOp = createUngroupedOp(log, rewriter, op, sparseOperands, dataOperands, dataResultTypes);
    auto smOp = createUngroupedOp(log, rewriter, op, sparseOperands, smOperands, smResultTypes);
    auto seTableOp = createUngroupedOp(log, rewriter, op, sparseOperands, seTableOperands, seTableResultTypes);

    auto dataResult = dataOp->getResult(0);
    auto smResult = (smOp != nullptr) ? smOp->getResult(0) : nullptr;
    auto seTableResult = (seTableOp != nullptr) ? seTableOp->getResult(0) : nullptr;
    auto isWeightsVar = (isWeights.size() > 0) ? (isWeights[0] != nullptr) : false;
    auto sparsityCompressionAttr = (sparsityCompressions.size() > 0) ? sparsityCompressions[0] : nullptr;
    auto seAttr = (seAttrs.size() > 0) ? seAttrs[0] : nullptr;

    rewriter.replaceOpWithNewOp<VPUIP::GroupSparseBufferOp>(op, dataResult, smResult, seTableResult, isWeightsVar,
                                                            sparsityCompressionAttr, seAttr);
}

//
// RemoveGroupUngroup
//

class RemoveGroupUngroupRewriter final : public mlir::OpRewritePattern<VPUIP::GroupSparseBufferOp> {
public:
    RemoveGroupUngroupRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::GroupSparseBufferOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::GroupSparseBufferOp groupOp,
                                        mlir::PatternRewriter& rewriter) const final {
        std::vector<mlir::Operation*> candidates;
        for (auto user : groupOp->getUsers()) {
            if (mlir::isa<VPUIP::UngroupSparseBufferOp>(user)) {
                continue;
            }
            candidates.push_back(user);
        }

        const auto getSparseValues = [](mlir::ValueRange values) -> SmallVector<mlir::Value> {
            SmallVector<mlir::Value> sparseValues;
            for (auto value : values) {
                if (mlir::isa<vpux::VPUIP::SparseBufferType>(value.getType())) {
                    sparseValues.push_back(value);
                }
            }
            return sparseValues;
        };
        if (candidates.size() > 0) {
            for (auto user : candidates) {
                const auto sparseOperands = getSparseValues(user->getOperands());
                const auto sparseResults = getSparseValues(user->getResults());
                if (sparseOperands.empty() && sparseResults.empty()) {
                    continue;
                }

                rewriter.setInsertionPointAfter(user);
                ungroupOperation(_log, rewriter, user, sparseOperands, sparseResults);
            }
        }

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

private:
    Logger _log;
};

}  // namespace

void vpux::VPUIP::registerUngroupSparseBufferRewriters(vpux::RewriterRegistry& registry, Logger& log) {
    registry.registerRewriter<RemoveGroupUngroupRewriter>("ungroup-sparse-buffer", log);
}
