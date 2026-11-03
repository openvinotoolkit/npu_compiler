//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/weights_separation.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/dialect.hpp"
#include "vpux/compiler/dialect/core/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <llvm/ADT/DenseMapInfo.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/IRMapping.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_CONCATHOSTCONSTS
#define GEN_PASS_DEF_CONCATHOSTCONSTS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

//
// ConcatHostConstsPass
//

class ConcatHostConstsPass final : public VPU::impl::ConcatHostConstsBase<ConcatHostConstsPass> {
public:
    explicit ConcatHostConstsPass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    struct ObfuscationInfo {
        mlir::DenseSet<mlir::func::CallOp> callOpsToUpdate;
        mlir::DenseMap<size_t, size_t> argToFoldedOffset;
    };

private:
    void safeRunOnModule() final;
    Logger _log;
};

namespace {

// Prepare the per-argument rewrite data in the same order used for the shared blob.
// The caller keeps the original argument indices and the matching byte offsets as
// separate arrays so the existing obfuscation code can consume them directly
std::pair<SmallVector<size_t>, SmallVector<int64_t>> prepareInputSlices(
        const ConcatHostConstsPass::ObfuscationInfo& info) {
    std::pair<SmallVector<size_t>, SmallVector<int64_t>> indicesAndOffsets;
    auto& indices = indicesAndOffsets.first;
    auto& offsets = indicesAndOffsets.second;
    indices.reserve(info.argToFoldedOffset.size());
    offsets.reserve(info.argToFoldedOffset.size());
    for (const auto& [argIdx, foldedOffset] : info.argToFoldedOffset) {
        indices.push_back(argIdx);
        offsets.push_back(checked_cast<int64_t>(foldedOffset));
    }
    return indicesAndOffsets;
}

// Rebuild the call operand list after extraction. The rewritten callee receives the full
// concatenated constant as one argument, so the old constant operands must be removed from
// the call and replaced with the single folded blob. We drop operands by original arg index
// instead of truncating a prefix because the extracted constants can come from arbitrary
// positions in the original signature.
SmallVector<mlir::Value> getUpdatedCallOperands(mlir::func::CallOp callOp, ArrayRef<size_t> inputIndices,
                                                mlir::Value foldedConst) {
    const auto nonConstOrigArgumentCount = callOp->getNumOperands() - inputIndices.size();
    SmallVector<mlir::Value> newOperands(callOp->getOperands().begin(),
                                         callOp->getOperands().begin() + nonConstOrigArgumentCount);
    newOperands.push_back(foldedConst);

    return newOperands;
}

// Only fold constants that are exclusively consumed by calls. A host-side use would
// need the original declaration to stay alive, so including it here would make the
// subsequent erase invalidate that use
bool isCallOnlyConstant(Const::DeclareOp constOp) {
    return !constOp->getUses().empty() && llvm::all_of(constOp->getUses(), [](auto& use) {
        return mlir::isa<mlir::func::CallOp>(use.getOwner());
    });
}

}  // namespace

// Concatenates all host constants into a single arith constant
mlir::arith::ConstantOp foldConstants(mlir::Location loc, mlir::func::FuncOp funcOp,
                                      std::vector<Const::DeclareOp>& constOps) {
    mlir::OpBuilder builder(funcOp);

    auto lastConstOp = constOps.back();
    builder.setInsertionPointAfter(lastConstOp);

    size_t totalBuffSize = 0;
    for (auto constOp : constOps) {
        const auto content = constOp.getContent();
        const auto contentType = content.getType();
        totalBuffSize += checked_cast<size_t>(contentType.getTotalAllocSize().count());
    }
    std::vector<char> buffer(totalBuffSize);
    size_t offset = 0;
    for (auto constOp : constOps) {
        const auto content = constOp.getContent();
        const auto contentType = content.getType();
        auto bufSize = checked_cast<size_t>(contentType.getTotalAllocSize().count());
        content.copyTo(MutableArrayRef(buffer.data() + offset, bufSize));
        offset += bufSize;
    }
    auto rankedTensorType = mlir::RankedTensorType::get(ShapeRef({checked_cast<int32_t>(totalBuffSize)}),
                                                        getInt8Type(funcOp.getContext()));
    const auto denseAttr = Const::createConstContent(rankedTensorType, buffer);
    return mlir::arith::ConstantOp::create(builder, loc, rankedTensorType, denseAttr);
}

void ConcatHostConstsPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto mainFuncOp = net::getFromModule(moduleOp).second;

    std::vector<Const::DeclareOp> extractedConstOps = [&]() {
        auto infoOpt = getCachedAnalysis<VPU::WeightsSeparationInfo>();
        VPUX_THROW_UNLESS(infoOpt.has_value(), "VPU::WeightsSeparationInfo analysis must be cached");
        const auto& info = infoOpt->get();

        auto splits = info.getCollectedSplits();

        std::vector<Const::DeclareOp> data;

        mainFuncOp->walk([&](Const::DeclareOp constOp) {
            auto constAttr = constOp.getContentAttr();
            auto baseContent = constAttr.getBaseContent();
            auto it = llvm::find_if(splits, [&](const VPU::TransformationsSplit& split) {
                return split.getContentAttr().getBaseContent() == baseContent;
            });
            if (it != splits.end() && isCallOnlyConstant(constOp)) {
                data.push_back(constOp);
            }
        });

        return data;
    }();

    if (extractedConstOps.empty()) {
        _log.debug("No host constants to concatenate");
        return;
    }

    // Sort by block order so the folded blob follows the same constant order that exists
    // after extract-weights. This keeps the byte offsets deterministic, lets us insert the
    // builder after the last op, and makes foldedOffset stable across callees
    llvm::sort(extractedConstOps, [](auto& lhs, auto& rhs) {
        return lhs->isBeforeInBlock(rhs);
    });

    mlir::DenseMap<mlir::func::FuncOp, ObfuscationInfo> obfuscateInfo;
    size_t foldedOffset = 0;
    for (auto op : extractedConstOps) {
        const auto content = op.getContent();
        const auto contentType = content.getType();
        const auto contentSize = checked_cast<size_t>(contentType.getTotalAllocSize().count());

        // Every use of the same extracted constant in a specific callee gets the same
        // foldedOffset. Different callees can observe different subsets, but the offset for a
        // given original operand stays identical because it is derived from the shared blob
        // order above, not from the local subset
        for (auto& use : op->getUses()) {
            if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(use.getOwner()); callOp != nullptr) {
                const auto argIdx = checked_cast<size_t>(use.getOperandNumber());
                auto calleeFunc = moduleOp.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
                assert(calleeFunc != nullptr && "CallOp must have callee func");
                obfuscateInfo[calleeFunc].callOpsToUpdate.insert(callOp);
                auto& argToFoldedOffset = obfuscateInfo[calleeFunc].argToFoldedOffset;
                const auto [it, inserted] = argToFoldedOffset.try_emplace(argIdx, foldedOffset);
                VPUX_THROW_UNLESS(inserted || it->second == foldedOffset,
                                  "Argument '{0}' of function '{1}' maps to multiple folded constant offsets: '{2}' "
                                  "and '{3}'",
                                  argIdx, calleeFunc.getSymName(), it->second, foldedOffset);

                _log.debug("Marking operand {0} of call op {1} for obfuscation at folded constant offset {2}", argIdx,
                           callOp, foldedOffset);
            }
        }

        foldedOffset += contentSize;
    }

    mlir::OpBuilder builder(mainFuncOp);
    auto foldedConstOp =
            foldConstants(appendLoc(mainFuncOp.getLoc(), "obfuscated_outputs"), mainFuncOp, extractedConstOps);

    for (auto [funcOp, info] : obfuscateInfo) {
        auto [inputIndices, inputOffsets] = prepareInputSlices(info);

        VPU::obfuscateInputs(_log.nest(), appendLoc(funcOp.getLoc(), "obfuscated_inputs"), funcOp, inputIndices,
                             inputOffsets, foldedConstOp.getType(),
                             [](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                                ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes) {
                                 return VPU::SliceOp::create(builder, loc, input, offsets, sizes);
                             });

        for (auto callOp : info.callOpsToUpdate) {
            auto newOperands = getUpdatedCallOperands(callOp, inputIndices, foldedConstOp);
            builder.setInsertionPoint(callOp);
            auto newCallOp = mlir::func::CallOp::create(builder, appendLoc(callOp->getLoc(), "_updated"),
                                                        callOp.getCalleeAttr(), callOp->getResultTypes(), newOperands);
            callOp->replaceAllUsesWith(newCallOp);
            callOp->erase();
        }
    }

    for (auto constOp : extractedConstOps) {
        constOp->erase();
    }
}

//
// createConcatHostConstsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createConcatHostConstsPass(Logger log) {
    return std::make_unique<ConcatHostConstsPass>(log);
}
