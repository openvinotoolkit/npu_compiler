//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_unroll_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"

#include <mlir/Dialect/Affine/Analysis/Utils.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Utils/Utils.h>
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"

namespace vpux::VPU {
#define GEN_PASS_DECL_UNROLLSCFLOOP
#define GEN_PASS_DEF_UNROLLSCFLOOP
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

struct LoopInfo {
    mlir::Operation* loopOp;
    vpux::Dim dim;
    int64_t unrollFactor;
    int64_t depth;
};

//
// UnrollSCFLoopPass
//
class UnrollSCFLoopPass final : public VPU::impl::UnrollSCFLoopBase<UnrollSCFLoopPass> {
public:
    explicit UnrollSCFLoopPass(StringRef loopUnrollFactor, Logger log): _loopUnrollFactor(loopUnrollFactor) {
        Base::initLogger(log, Base::getArgumentName());
    }

    SmallVector<int64_t> parseUnrollFactorsStr(llvm::StringRef unrollFactorStr) {
        SmallVector<int64_t> factors;
        SmallVector<llvm::StringRef, 4> tokens;
        unrollFactorStr.split(tokens, ',');
        for (auto token : tokens) {
            token = token.trim();  // Remove whitespace
            int64_t value;
            if (token.getAsInteger(10, value)) {
                _log.error("Failed to parse unroll factor from string: {0}", token);
                return {};
            }
            factors.push_back(value);
        }
        return factors;
    }

    int64_t getUniqueNumber() {
        return ++_counter;
    }

private:
    int64_t _counter = 0;
    std::string _loopUnrollFactor;
    void collectForOpsInnerToOuter(mlir::Operation* rootOp, SmallVector<LoopInfo>& loops);
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) override;
    void safeRunOnModule() final;
};

unsigned getNestingDepth(mlir::Operation* op) {
    mlir::Operation* currOp = op;
    unsigned depth = 0;
    while ((currOp = currOp->getParentOp())) {
        if (mlir::isa<mlir::scf::ForOp>(currOp)) {
            depth++;
        }
    }
    return depth;
}

void UnrollSCFLoopPass::collectForOpsInnerToOuter(mlir::Operation* rootOp, SmallVector<LoopInfo>& loops) {
    // Map to store loops by their nesting depth
    llvm::DenseMap<unsigned, SmallVector<mlir::scf::ForOp, 4>> depthMap;
    unsigned maxDepth = 0;

    // Collect all ForOps and group by depth
    rootOp->walk([&](mlir::scf::ForOp forOp) {
        unsigned depth = getNestingDepth(forOp.getOperation());
        depthMap[depth].push_back(forOp);
        maxDepth = std::max(maxDepth, depth);
    });

    // Add loops from innermost (highest depth) to outermost (lowest depth)
    for (int depth = maxDepth; depth >= 0; --depth) {
        for (auto forOp : depthMap[depth]) {
            forOp->setAttr("id",
                           mlir::IntegerAttr::get(mlir::IntegerType::get(forOp.getContext(), 64), getUniqueNumber()));
            loops.push_back({forOp.getOperation(), vpux::Dim(0), 1, static_cast<int64_t>(depth)});
        }
    }
}

mlir::scf::ForOp findParentForOpFromOffset(mlir::OpFoldResult offset) {
    if (auto offsetValue = mlir::dyn_cast_or_null<mlir::Value>(offset)) {
        if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(offsetValue)) {
            if (auto scfForOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(blockArg.getOwner()->getParentOp())) {
                return scfForOp;
            }
        } else {
            AffineChainUtils affineUtils;
            auto opChain = affineUtils.collectAffineOpsChain(offsetValue);
            for (auto op : opChain) {
                for (auto operand : op->getOperands()) {
                    if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(operand)) {
                        if (auto scfForOp =
                                    mlir::dyn_cast_or_null<mlir::scf::ForOp>(blockArg.getOwner()->getParentOp())) {
                            return scfForOp;
                        }
                    }
                }
            }
        }
    }
    return nullptr;
}

void mapUnrollFactorToLoop(mlir::func::FuncOp funcOp, const SmallVector<int64_t>& unrollFactor,
                           SmallVector<LoopInfo>& loops) {
    funcOp.walk([&](mlir::scf::ForOp forOp) {
        auto insertSliceOps = forOp.getOps<mlir::tensor::InsertSliceOp>();
        for (auto insertSliceOp : insertSliceOps) {
            for (auto [idx, offset] : llvm::enumerate(insertSliceOp.getMixedOffsets())) {
                if (unrollFactor[idx] > 1) {
                    mlir::scf::ForOp parentLoop = findParentForOpFromOffset(offset);
                    if (parentLoop != nullptr) {
                        for (auto& loop : loops) {
                            if (loop.loopOp == parentLoop.getOperation()) {
                                loop.unrollFactor = unrollFactor[idx];
                                loop.dim = vpux::Dim(idx);
                                break;
                            }
                        }
                    }
                }
            }
        }
    });
}

/**
 * @brief Fuses sibling unrolled loops within the same block that share the same ID attribute.
 *
 * This function processes all blocks in the given function operation to find and merge
 * sibling SCF for loops that have been unrolled. Loops are grouped by their "id" attribute
 * and categorized as either regular unrolled loops or residual loops (marked with "residual"
 * attribute). Within each group, sibling loops are fused together to reduce the number of
 * loop operations and improve code structure.
 *
 * loop with attribute unrolled with a unroll factor specified as input while loop with attribute
 * residual are the loops that handle the remaining iterations with unroll factor of 1.
 */
mlir::LogicalResult fuseSiblingUnrolledLoops(mlir::func::FuncOp funcOp) {
    mlir::OpBuilder builder(funcOp);
    mlir::IRRewriter rewriter(builder);

    auto mergeUnrolledLoops = [&](SmallVector<mlir::scf::ForOp>& unrolledLoops, mlir::IRRewriter& rewriter,
                                  bool residualLoop = false) {
        if (unrolledLoops.size() <= 1) {
            return;
        }

        mlir::scf::ForOp targetForOp = unrolledLoops.front();
        for (size_t idx = 1; idx < unrolledLoops.size(); idx++) {
            targetForOp = fuseSiblingForLoops(targetForOp, unrolledLoops[idx], rewriter, residualLoop);
        }

        moveAffineArithOpsEarly(*targetForOp.getBody());
    };

    llvm::SetVector<mlir::Block*> uniqueBlocks;
    SmallVector<SmallVector<mlir::scf::ForOp>> loopsToProcess;
    funcOp->walk([&](mlir::scf::ForOp loop) {
        auto currentBlock = loop->getBlock();
        if (uniqueBlocks.contains(currentBlock)) {
            return;
        }

        uniqueBlocks.insert(loop->getBlock());
        auto forOpsIt = loop->getBlock()->getOps<mlir::scf::ForOp>();
        SmallVector<mlir::scf::ForOp> loopsInBlock;
        for (auto forOp : forOpsIt) {
            loopsInBlock.push_back(forOp);
        }
        loopsToProcess.push_back(loopsInBlock);
    });

    for (auto allLoopsInBlocks : llvm::make_early_inc_range(loopsToProcess)) {
        std::map<int64_t, SmallVector<mlir::scf::ForOp>> loopGroups;

        // Group them into separate vectors based on their id
        for (auto forOp : allLoopsInBlocks) {
            auto idAttr = forOp->getAttrOfType<mlir::IntegerAttr>("id");
            if (idAttr == nullptr) {
                return errorAt(forOp->getLoc(), "Loop has no id attribute");
            }
            loopGroups[idAttr.getInt()].push_back(forOp);
        }

        // process each group
        for (auto [idx, loopsWithSameId] : loopGroups) {
            SmallVector<mlir::scf::ForOp> unrolledLoops;
            SmallVector<mlir::scf::ForOp> residualLoops;

            for (auto loop : loopsWithSameId) {
                if (loop->hasAttr("residual")) {
                    residualLoops.push_back(loop);
                } else {
                    unrolledLoops.push_back(loop);
                }
            }

            mergeUnrolledLoops(unrolledLoops, rewriter);
            mergeUnrolledLoops(residualLoops, rewriter, true);
        }
        loopGroups.clear();
    }
    return mlir::success();
}

/**
 * @brief Processes and merges unrolled SCF loops within a function by analyzing tile dimensions
 * and applying unroll factors to optimize loop operations.
 *
 * This function walks through all SCF ForOp operations in the given function and processes
 * loops that contain tensor::InsertSliceOp operations. For each loop, it:
 * - Analyzes the mixed offsets of insert slice operations to identify tile dimensions
 * - Determines unroll factors for each dimension based on parent loop attributes
 * - Creates TileDimensionInfo structures containing loop IDs, dimensions, and unroll settings
 * - Sorts tile dimensions by depth (outermost to innermost) based on loop IDs
 * - Merges unrolled operations using the collected tile dimension information
 *
 * @param funcOp The MLIR function operation to process for loop unrolling
 * @param unrollFactor List of unroll factors to apply to different loop dimensions
 * @return mlir::LogicalResult Success if all loops were processed successfully,
 *         failure if any merge operation failed or was interrupted
 */
mlir::LogicalResult processUnrolledLoops(mlir::func::FuncOp funcOp, const SmallVector<int64_t>& unrollFactor) {
    auto walkResult = funcOp.walk([&](mlir::scf::ForOp forOp) {
        SmallVector<TileDimensionInfo> tileDimensionsInfo;
        auto insertSliceOps = forOp.getOps<mlir::tensor::InsertSliceOp>();
        auto numInsertSliceOps = std::distance(insertSliceOps.begin(), insertSliceOps.end());
        if (numInsertSliceOps == 0) {
            return mlir::WalkResult::advance();
        }

        auto firstSliceOp = *insertSliceOps.begin();
        for (auto [idx, offset] : llvm::enumerate(firstSliceOp.getMixedOffsets())) {
            auto constOffset = getConstantIntValue(offset);
            if (constOffset.has_value()) {
                continue;
            }

            auto parentLoop = findParentForOpFromOffset(offset);
            if (parentLoop != nullptr) {
                auto dimUnrollFactor = parentLoop->hasAttr("residual") ? 1 : unrollFactor[idx];
                auto loopId = parentLoop->getAttrOfType<mlir::IntegerAttr>("id");
                TileDimensionInfo dimInfo;
                dimInfo.id = loopId.getInt();
                dimInfo.dimension = vpux::Dim(idx);
                dimInfo.numBlocks = dimUnrollFactor;
                dimInfo.isUnrolled = (dimUnrollFactor > 1);
                tileDimensionsInfo.emplace_back(dimInfo);
            }
        }

        if (tileDimensionsInfo.empty()) {
            return mlir::WalkResult::advance();
        }

        // sort tileDimensionsInfo based on depth from outermost to innermost
        std::sort(tileDimensionsInfo.begin(), tileDimensionsInfo.end(),
                  [](const TileDimensionInfo& a, const TileDimensionInfo& b) {
                      return a.id > b.id;
                  });

        if (mlir::failed(vpux::VPU::mergeUnrolledOperations(forOp, tileDimensionsInfo))) {
            assert(false && "Failed to merge unrolled operations");
            return mlir::WalkResult::interrupt();
        }

        return mlir::WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
        return mlir::failure();
    }

    return mlir::success();
}

/**
 * @brief Unrolls SCF loops by specified factors and sets corresponding attributes on the resulting operations.
 *
 * This function processes a collection of loop information structures, performing loop unrolling
 * for loops with unroll factors greater than 1. After unrolling, it marks the unrolled loop
 * with an "unrolled" attribute and identifies any residual sibling loops that handle remaining
 * iterations, marking them with a "residual" attribute.
 *
 * @param builder MLIR OpBuilder used for creating attributes and managing the IR
 * @param loops Vector of LoopInfo structures containing loop operations and their unroll factors
 *
 * @return mlir::LogicalResult Returns success if all loop unrolling operations complete successfully,
 *         failure if any unrolling operation fails
 *
 */
mlir::LogicalResult unrollLoopsAndSetAttributes(mlir::OpBuilder& builder, SmallVector<LoopInfo>& loops) {
    for (auto& loopInfo : loops) {
        if (loopInfo.unrollFactor == 1) {
            continue;
        }

        auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(loopInfo.loopOp);
        auto result = mlir::loopUnrollByFactor(forOp, loopInfo.unrollFactor);
        if (mlir::failed(result)) {
            return mlir::failure();
        }
        forOp->setAttr("unrolled", builder.getUnitAttr());

        auto loopId = forOp->getAttrOfType<mlir::IntegerAttr>("id");
        mlir::Block* block = forOp->getBlock();
        auto forOps = block->getOps<mlir::scf::ForOp>();
        mlir::scf::ForOp siblingLoop = nullptr;
        for (auto forOp : forOps) {
            auto id = forOp->getAttrOfType<mlir::IntegerAttr>("id");
            if ((id == loopId) && !forOp->hasAttr("unrolled")) {
                if (siblingLoop != nullptr) {
                    return errorAt(forOp->getLoc(), "Multiple sibling loops found for loop with id {0}",
                                   loopId.getInt());
                }
                siblingLoop = forOp;
            }
        }

        if (siblingLoop != nullptr) {
            siblingLoop->setAttr("residual", builder.getUnitAttr());
        }
        moveAffineArithOpsEarly(*forOp.getBody());
    }

    return mlir::success();
}

void cleanupAttributes(mlir::func::FuncOp funcOp) {
    funcOp.walk([&](mlir::scf::ForOp forOp) {
        if (forOp->hasAttr("id")) {
            forOp->removeAttr("id");
        }

        if (forOp->hasAttr("unrolled")) {
            forOp->removeAttr("unrolled");
        }

        if (forOp->hasAttr("residual")) {
            forOp->removeAttr("residual");
        }
    });
}

mlir::LogicalResult UnrollSCFLoopPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (loopUnrollFactor.hasValue()) {
        _loopUnrollFactor = loopUnrollFactor.getValue();
    }

    return mlir::success();
}

void UnrollSCFLoopPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    net::NetworkInfoOp netInfoOp;
    mlir::func::FuncOp mainFuncOp;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfoOp, mainFuncOp);
    mlir::OpBuilder builder(mainFuncOp);

    if (_loopUnrollFactor.empty()) {
        _log.trace("Unroll factor not specified or invalid. Skipping unroll scf pass");
        signalPassFailure();
        return;
    }

    auto unrollFactorsVec = parseUnrollFactorsStr(_loopUnrollFactor);
    if (unrollFactorsVec.empty()) {
        _log.trace("Unroll factor not specified or invalid. Skipping unroll scf pass");
        return;
    }

    // Check unroll factor constraints using LLVM utilities
    auto numDimsToUnroll = llvm::count_if(unrollFactorsVec, [](auto factor) {
        return factor > 1;
    });

    if (numDimsToUnroll == 0) {
        _log.trace("All unroll factors are <= 1. Skipping unroll scf pass");
        return;
    }

    if (numDimsToUnroll > 2) {
        _log.error("Unsupported unroll config. Unroll scf pass failed");
        signalPassFailure();
        return;
    }

    /* Collect all loops in the function from innermost to outermost
       This is to ensure that the innermost loops are unrolled first and any
       outer loops that need to be unrolled are unrolled after that
    */
    SmallVector<LoopInfo> loopInfoVector;
    collectForOpsInnerToOuter(mainFuncOp.getOperation(), loopInfoVector);
    if (loopInfoVector.empty()) {
        _log.trace("No loops found. Skipping unroll scf pass");
        return;
    }

    mapUnrollFactorToLoop(mainFuncOp, unrollFactorsVec, loopInfoVector);

    if (mlir::failed(unrollLoopsAndSetAttributes(builder, loopInfoVector))) {
        _log.trace("Failed to unroll loops and set attributes");
        signalPassFailure();
        return;
    }

    if (mlir::failed(fuseSiblingUnrolledLoops(mainFuncOp))) {
        _log.trace("Failed to fuse sibling unrolled loops");
        signalPassFailure();
        return;
    }

    if (mlir::failed(processUnrolledLoops(mainFuncOp, unrollFactorsVec))) {
        _log.trace("Failed to process unrolled loops");
        signalPassFailure();
        return;
    }

    cleanupAttributes(mainFuncOp);
}
}  // namespace

//
// createUnrollSCFLoopPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createUnrollSCFLoopPass(StringRef loopUnrollFactor, Logger log) {
    return std::make_unique<UnrollSCFLoopPass>(loopUnrollFactor, log);
}
