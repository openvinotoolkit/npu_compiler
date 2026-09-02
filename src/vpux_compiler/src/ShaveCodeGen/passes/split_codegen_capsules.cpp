//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Pass/Pass.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_SPLITCODEGENCAPSULES
#define GEN_PASS_DEF_SPLITCODEGENCAPSULES
#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {

//
// SplitCodeGenCapsulesPass
//

class SplitCodeGenCapsulesPass final : public ShaveCodeGen::impl::SplitCodeGenCapsulesBase<SplitCodeGenCapsulesPass> {
public:
    explicit SplitCodeGenCapsulesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    mlir::LogicalResult runOnCapsule(IE::CodeGenCapsuleOp capsule, mlir::OpBuilder& builder);
    void safeRunOnFunc() final;
};

mlir::LogicalResult SplitCodeGenCapsulesPass::runOnCapsule(IE::CodeGenCapsuleOp capsule, mlir::OpBuilder& builder) {
    auto* capsuleBlock = capsule.getBody();

    auto yieldOp = mlir::cast<IE::CGCYieldOp>(capsuleBlock->getTerminator());

    // Map values inside the capsule block to their external counterparts:
    //   - block args map to capsule operands
    //   - KernelRegion results map to new capsule results
    mlir::IRMapping mapping;
    for (auto [arg, operand] : llvm::zip(capsuleBlock->getArguments(), capsule->getOperands())) {
        mapping.map(arg, operand);
    }

    // Map values to indices in the CGCYieldOp operand list.
    llvm::DenseMap<mlir::Value, unsigned> yieldPositionMap;
    for (auto [idx, yieldedVal] : llvm::enumerate(yieldOp->getOperands())) {
        yieldPositionMap[yieldedVal] = static_cast<unsigned>(idx);
    }

    // Insert new capsules sequentially before the original capsule so that
    // producers always precede consumers in IR order.
    builder.setInsertionPoint(capsule);

    // Perform a single forward pass over the capsule block (topological order is guaranteed).
    // Scalar ops are rematerialized outside the capsule; KernelRegions are moved externally to
    // new CodeGenCapsuleOps.
    bool failed = false;
    for (auto& op : *capsuleBlock) {
        if (mlir::isa<IE::CGCYieldOp>(&op)) {
            break;
        }

        mlir::TypeSwitch<mlir::Operation*>(&op)
                .Case<Shave::KernelRegionOp>([&](Shave::KernelRegionOp kr) {
                    // Translate the KernelRegion's operands to values defined outside the capsule.
                    llvm::SmallVector<mlir::Value> newInputs;
                    for (auto input : kr.getInputs()) {
                        auto mapped = mapping.lookupOrNull(input);
                        VPUX_THROW_UNLESS(mapped, "KernelRegion input '{0}' has no external mapping", input);
                        newInputs.push_back(mapped);
                    }

                    auto* bodyBlock = &kr.getBody().front();
                    auto krYield = mlir::cast<Shave::KernelRegionYieldOp>(bodyBlock->getTerminator());
                    // Snapshot yielded values before the yield op is erased.
                    auto krYieldedVals = llvm::to_vector(krYield->getOperands());

                    // Determine the result types of the new CodeGenCapsule.
                    //
                    // If the KernelRegion result feeds into the CGCYieldOp, use the original capsule's
                    // result type directly. CodeGenCapsule performs an implicit cast between the yield
                    // operand type and the result type, so no PermuteCast is needed. For intermediate
                    // results, use the existing KernelRegion output type.
                    llvm::SmallVector<mlir::Type> newResultTypes;
                    for (auto [resIdx, krResult] : llvm::enumerate(kr.getResults())) {
                        auto yieldIt = yieldPositionMap.find(krResult);
                        if (yieldIt != yieldPositionMap.end()) {
                            newResultTypes.push_back(capsule.getResultTypes()[yieldIt->second]);
                            continue;
                        }
                        newResultTypes.push_back(krResult.getType());
                    }

                    // Create the new CodeGenCapsuleOp at the current insertion point.
                    auto newCap = builder.create<IE::CodeGenCapsuleOp>(kr.getLoc(), newResultTypes, newInputs);

                    // Move the KernelRegion body into the new capsule.
                    auto* dummyBlock = &newCap.getContent().emplaceBlock();
                    bodyBlock->moveBefore(dummyBlock);
                    dummyBlock->erase();

                    // Replace the KernelRegion.Yield terminator with an IE.CGCYield.
                    // Use a local builder scoped to the body so the outer builder's
                    // insertion point is not disturbed.
                    mlir::OpBuilder bodyBuilder(krYield);
                    bodyBuilder.create<IE::CGCYieldOp>(krYield->getLoc(), krYieldedVals);
                    krYield.erase();

                    // Update mapping so subsequent KernelRegions can refer to the new capsule results.
                    for (auto [resIdx, krResult] : llvm::enumerate(kr.getResults())) {
                        mapping.map(krResult, newCap.getResult(resIdx));
                    }
                })
                .Case<mlir::tensor::DimOp>([&](mlir::tensor::DimOp dimOp) {
                    auto resolvedSrc = mapping.lookupOrNull(dimOp.getSource());
                    VPUX_THROW_UNLESS(resolvedSrc, "tensor.dim source has no external mapping");

                    // If the mapped operand has a different DimsOrder update the rank operand.
                    auto constOp = dimOp.getIndex().getDefiningOp<mlir::arith::ConstantIndexOp>();
                    if (!constOp) {
                        mlir::emitError(dimOp.getLoc())
                                << "tensor.dim index is expected to be an arith.constant index op";
                        failed = true;
                        return;
                    }
                    int64_t dimIdx = constOp.value();

                    auto origNdType = mlir::dyn_cast<vpux::NDTypeInterface>(dimOp.getSource().getType());
                    auto newNdType = mlir::dyn_cast<vpux::NDTypeInterface>(resolvedSrc.getType());
                    if (origNdType && newNdType) {
                        auto origOrder = origNdType.getDimsOrder();
                        auto newOrder = newNdType.getDimsOrder();
                        if (!origOrder.isIdentity()) {
                            mlir::emitError(dimOp.getLoc())
                                    << "CodeGenCapsule does not match the expected canonical form: "
                                       "tensor.dim source has non-identity DimsOrder";
                            failed = true;
                            return;
                        }
                        if (origOrder != newOrder) {
                            auto logicalDim = newOrder.dimAt(static_cast<size_t>(dimIdx));
                            dimIdx = static_cast<int64_t>(logicalDim.ind());
                        }
                    }

                    auto newDimVal = builder.create<mlir::arith::ConstantIndexOp>(dimOp.getLoc(), dimIdx);
                    auto newDimOp = builder.create<mlir::tensor::DimOp>(dimOp.getLoc(), resolvedSrc, newDimVal);
                    mapping.map(dimOp.getResult(), newDimOp.getResult());
                })
                .Case<mlir::arith::ConstantOp>([&](mlir::arith::ConstantOp constOp) {
                    builder.clone(*constOp, mapping);
                })
                .Default([&](mlir::Operation* unexpectedOp) {
                    mlir::emitError(unexpectedOp->getLoc())
                            << "unexpected op '" << unexpectedOp->getName()
                            << "' in CodeGenCapsule: expected only Shave.KernelRegion, IE.CGCYield, "
                               "tensor.dim, or arith.constant";
                    failed = true;
                });

        if (failed) {
            return mlir::failure();
        }
    }

    // Replace all uses of the original capsule's results and erase it.
    for (auto [origResult, yieldedVal] : llvm::zip(capsule.getResults(), yieldOp->getOperands())) {
        auto replacement = mapping.lookupOrNull(yieldedVal);
        VPUX_THROW_UNLESS(replacement, "Yielded value '{0}' has no external mapping", yieldedVal);
        origResult.replaceAllUsesWith(replacement);
    }
    capsule.erase();
    return mlir::success();
}

void SplitCodeGenCapsulesPass::safeRunOnFunc() {
    auto func = getOperation();
    mlir::OpBuilder builder(func.getContext());

    auto capsules = llvm::to_vector(func.getOps<IE::CodeGenCapsuleOp>());
    for (auto capsule : capsules) {
        if (mlir::failed(runOnCapsule(capsule, builder))) {
            signalPassFailure();
            return;
        }
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createSplitCodeGenCapsulesPass(Logger log) {
    return std::make_unique<SplitCodeGenCapsulesPass>(log);
}
