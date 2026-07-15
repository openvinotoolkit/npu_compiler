//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/weights_separation.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

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
        mlir::DenseSet<size_t> argsToObfuscate;
    };

private:
    void safeRunOnModule() final;
    Logger _log;
};

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
    return builder.create<mlir::arith::ConstantOp>(loc, rankedTensorType, denseAttr);
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
            if (it != splits.end()) {
                data.push_back(constOp);
            }
        });

        return data;
    }();

    if (extractedConstOps.empty()) {
        _log.debug("No host constants to concatenate");
        return;
    }

    mlir::DenseMap<mlir::func::FuncOp, ObfuscationInfo> obfuscateInfo;
    for (auto op : extractedConstOps) {
        for (auto& use : op->getUses()) {
            if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(use.getOwner()); callOp != nullptr) {
                unsigned argIdx = use.getOperandNumber();
                auto calleeFunc = moduleOp.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
                assert(calleeFunc != nullptr && "CallOp must have callee func");
                obfuscateInfo[calleeFunc].callOpsToUpdate.insert(callOp);
                obfuscateInfo[calleeFunc].argsToObfuscate.insert(argIdx);

                _log.debug("Marking operand {0} of call op {1} for obfuscation", argIdx, callOp);
            }
        }
    }

    mlir::OpBuilder builder(mainFuncOp);

    // Sort in topological order to insert a builder after the last op
    llvm::sort(extractedConstOps, [](auto& lhs, auto& rhs) {
        return lhs->isBeforeInBlock(rhs);
    });
    auto foldedConstOp =
            foldConstants(appendLoc(mainFuncOp.getLoc(), "obfuscated_outputs"), mainFuncOp, extractedConstOps);

    for (auto [funcOp, info] : obfuscateInfo) {
        SmallVector<size_t> argsVec(info.argsToObfuscate.begin(), info.argsToObfuscate.end());
        VPU::obfuscateInputs(
                _log.nest(), appendLoc(funcOp.getLoc(), "obfuscated_inputs"), funcOp, argsVec,
                [](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input, ArrayRef<int64_t> offsets,
                   ArrayRef<int64_t> sizes) {
                    return builder.create<VPU::SliceOp>(loc, input, offsets, sizes);
                },
                /*obfuscateSingleInput=*/true);

        for (auto callOp : info.callOpsToUpdate) {
            // After obfuscation the last funcOp's argument is the concatenated constant and all the original arguments
            // are before it
            auto offset = funcOp.getNumArguments() == 0 ? 0 : funcOp.getNumArguments() - 1;
            SmallVector<mlir::Value> newOperands(callOp->getOperands().begin(), callOp->getOperands().begin() + offset);
            newOperands.push_back(foldedConstOp);
            builder.setInsertionPoint(callOp);
            auto newCallOp =
                    builder.create<mlir::func::CallOp>(appendLoc(callOp->getLoc(), "_updated"), callOp.getCalleeAttr(),
                                                       callOp->getResultTypes(), newOperands);
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
