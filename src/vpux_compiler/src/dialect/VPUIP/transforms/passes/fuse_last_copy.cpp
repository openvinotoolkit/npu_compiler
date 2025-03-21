//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/core/aliases_info.hpp"

#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_FUSELASTCOPY
#define GEN_PASS_DEF_FUSELASTCOPY
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

template <typename OpTy>
mlir::Value replaceCastWith(mlir::Operation* op, mlir::Value sourceRoot, mlir::Type outType, mlir::Value inputValue) {
    mlir::OpBuilder builder(op);
    builder.setInsertionPoint(sourceRoot.getDefiningOp());
    auto newOperation = builder.create<OpTy>(op->getLoc(), outType, inputValue);
    return newOperation.getResult();
};

void fuseLastCopy(VPUIP::CopyOp copyOp, const AliasesInfo& aliasesInfo, Logger log) {
    log.trace("fuseLastCopy: Copy at {0}", copyOp->getLoc());
    auto nestedLogger = log.nest();

    auto inSourceMemory = copyOp.getInput().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
    auto outSourceMemory = copyOp.getOutput().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
    if (inSourceMemory != outSourceMemory) {
        nestedLogger.trace("Cannot match because the copy is not within the same memory space");
        return;
    }

    auto sourceOp = copyOp.getInput().getDefiningOp();
    if (sourceOp == nullptr) {
        nestedLogger.trace("Cannot match: copy input is a block argument");
        return;
    }

    const auto sourceRoots = aliasesInfo.getRoots(copyOp.getInput());
    if (sourceRoots.size() != 1) {
        nestedLogger.trace("Cannot match: expected single source root for {0} but got {1}", copyOp.getInput(),
                           sourceRoots.size());
        return;
    }

    const auto sourceRoot = *sourceRoots.begin();
    if (sourceRoot == nullptr || mlir::isa<mlir::BlockArgument>(sourceRoot)) {
        nestedLogger.trace("Cannot match: input is a block argument");
        return;
    }
    const auto sourceRootOp = sourceRoot.getDefiningOp();
    if (!isBufAllocOp(sourceRootOp)) {
        nestedLogger.trace("Cannot match: input isn't allocate op but '{0}'", sourceRootOp->getName());
        return;
    }

    auto allRootAliases = aliasesInfo.getAllAliases(sourceRoot);
    for (auto alias : allRootAliases) {
        for (auto userOp : alias.getUsers()) {
            if (auto copyUserOp = mlir::dyn_cast<VPUIP::CopyOp>(userOp)) {
                if (copyUserOp != copyOp && copyUserOp.getOutputBuff().isa<mlir::BlockArgument>()) {
                    nestedLogger.trace("Cannot fuse when there are multiple output copy operations");
                    return;
                }
            }
        }
    }

    VPUIP::ConcatViewOp concatViewOp;
    auto newBuffer = copyOp.getOutputBuff();
    auto newOutput = copyOp.getInput();
    mlir::Operation* typeCastOp = nullptr;

    auto isCastOp = [](mlir::Operation* maybeTypeCastOp) {
        return mlir::isa_and_present<VPUIP::GenericReshapeOp, VPUIP::QuantizeCastOp, VPUIP::PermuteCastOp,
                                     VPUIP::ShapeCastOp>(maybeTypeCastOp);
    };

    if (sourceRoot.getType() != copyOp.getOutputBuff().getType()) {
        // check what operation changes the type
        typeCastOp = copyOp.getInput().getDefiningOp();

        if (typeCastOp == nullptr || std::distance(typeCastOp->getUsers().begin(), typeCastOp->getUsers().end()) != 1) {
            // skip if typeCastOp has multi users
            return;
        }

        auto preOfTypeCastOp = typeCastOp;
        while (isCastOp(preOfTypeCastOp)) {
            // clang-format off
            // We will make a OpTy(QuantizeCast/GenericReshape) over the output buffer and we will copy from CMX directly
            // to output buffer, and we will return the output buffer. After ConcatView and OpTy will be redundant.
            // To do this, reverse and apply the same cast operations to the output buffer.
            // Original IR:
            // Copy(CMX->DDR) -> CastOp1(!type1->!type2) -> CastOp2(!type2->!type3) -> Copy(DDR->DDR) to output_buffer
            // New IR:
            // output_buffer -> CastOp2(!type3 -> !type2) -> CastOp1(!type2 -> !type1) -> new "output buffer"
            // source (from CMX) -> Copy(CMX->DDR) new "output buffer"
            // clang-format on

            if (mlir::isa<VPUIP::GenericReshapeOp>(preOfTypeCastOp)) {
                newBuffer = replaceCastWith<VPUIP::GenericReshapeOp>(
                        preOfTypeCastOp, sourceRoot, preOfTypeCastOp->getOperand(0).getType(), newBuffer);
            } else if (mlir::isa<VPUIP::QuantizeCastOp>(preOfTypeCastOp)) {
                newBuffer = replaceCastWith<VPUIP::QuantizeCastOp>(preOfTypeCastOp, sourceRoot,
                                                                   preOfTypeCastOp->getOperand(0).getType(), newBuffer);
            } else if (auto permuteCastOp = mlir::dyn_cast<VPUIP::PermuteCastOp>(preOfTypeCastOp)) {
                // clang-format off
                // do the permute in output
                // Here we produce invalid operation since we swap I/O type while attributes are the same.
                // From test cases:
                // Original operation: VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs(%5 : memref<1x263169x11x1xf16, @DDR>) -> memref<1x11x1x263169xf16, #NHWC, @DDR>
                // Modified IR: VPUIP.PermuteCast {dst_order = #NHWC, mem_perm = #NCHW} inputs([[OUTPUT]] : memref<1x11x1x263169xf16, #NHWC, @DDR>) -> memref<1x263169x11x1xf16, @DDR>
                // You may see that for modified PermuteCast output type has #NCHW order, while dst_order #NHWC
                // This is possible due to lack of validation.
                // TODO: #-141102
                // clang-format on
                mlir::OpBuilder builder(permuteCastOp);
                builder.setInsertionPoint(sourceRoot.getDefiningOp());

                auto newPermuteCast = builder.create<VPUIP::PermuteCastOp>(
                        permuteCastOp.getLoc(), permuteCastOp->getOperand(0).getType(), newBuffer,
                        permuteCastOp.getDstOrderAttr(), permuteCastOp.getMemPermAttr());

                newBuffer = newPermuteCast.getResult();
            } else if (auto shapeCastOp = mlir::dyn_cast<VPUIP::ShapeCastOp>(preOfTypeCastOp)) {
                // check is simple ShapeCast
                if (shapeCastOp.getExplicitOutputShapes().has_value() ||
                    shapeCastOp.getExplicitOutputOffsets().has_value()) {
                    nestedLogger.trace("ShapeCast with explicit shapes and offsets not supported");
                    return;
                }

                auto newOutType = mlir::cast<NDTypeInterface>(shapeCastOp.getSource().getType());
                auto newOutShape = newOutType.getShape().raw();

                // do the shape cast in output
                mlir::OpBuilder builder(shapeCastOp);
                builder.setInsertionPoint(sourceRoot.getDefiningOp());
                auto newShapeCast = builder.create<VPUIP::ShapeCastOp>(
                        shapeCastOp.getLoc(), newOutType, newBuffer,
                        getIntArrayAttr(shapeCastOp.getContext(), newOutShape),
                        shapeCastOp.getExplicitOutputShapesAttr(), shapeCastOp.getExplicitOutputOffsetsAttr());
                newBuffer = newShapeCast.getResult();
            } else {
                VPUX_THROW("Unsupported cast operation: {0}", preOfTypeCastOp->getName());
            }

            if (!preOfTypeCastOp->hasOneUse()) {
                return;
            }
            preOfTypeCastOp = preOfTypeCastOp->getOperand(0).getDefiningOp();
        }

        // from CMX -> CopyOp[DDR] -> (ConcatViewOp) -> OpTy -> CopyOp[block-arg] -> return CopyOp
        // Output of this step:
        //                        CMX -> CopyOp[OpTy] -> return block-arg
        //   block-arg -> OpTy /

        concatViewOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(preOfTypeCastOp);
        if (concatViewOp != nullptr) {
            if (!concatViewOp.getOutput().hasOneUse()) {
                return;
            }
        } else if (VPUIP::isPureViewOp(preOfTypeCastOp)) {
            return;
        }

        newOutput = copyOp.getOutputBuff();
    }

    // Function outputs have to be an alias of the output buffer
    log.trace("Root of the copy operation input {0}", sourceRoot);
    log.trace("Reassign outputs from {0} to {1}", sourceRoot, newBuffer);

    for (auto& use : llvm::make_early_inc_range(sourceRoot.getUses())) {
        log.nest().trace("Got user {0}", use.getOwner()->getName());
        log.nest().trace("Reassign {0} to {1}", use.get(), newBuffer);
        use.set(newBuffer);
    }

    copyOp.replaceAllUsesWith(newOutput);
    copyOp->erase();

    auto preOfTypeCastOp = typeCastOp;
    while (isCastOp(preOfTypeCastOp)) {
        auto currOp = preOfTypeCastOp;
        preOfTypeCastOp = preOfTypeCastOp->getOperand(0).getDefiningOp();
        currOp->erase();
    }

    if (concatViewOp) {
        concatViewOp->erase();
    }

    if (sourceRootOp->use_empty()) {
        sourceRootOp->erase();
    }
}

//
// FuseLastCopy
//

class FuseLastCopyPass final : public VPUIP::impl::FuseLastCopyBase<FuseLastCopyPass> {
public:
    explicit FuseLastCopyPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void FuseLastCopyPass::safeRunOnFunc() {
    auto func = getOperation();

    func->walk([&](VPUIP::CopyOp op) {
        if (!op.getOutputBuff().isa<mlir::BlockArgument>()) {
            return;
        }

        auto& aliasInfo = getAnalysis<AliasesInfo>();
        fuseLastCopy(op, aliasInfo, _log);
    });
}

}  // namespace

//
// createFuseLastCopyPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createFuseLastCopyPass(Logger log) {
    return std::make_unique<FuseLastCopyPass>(log);
}
