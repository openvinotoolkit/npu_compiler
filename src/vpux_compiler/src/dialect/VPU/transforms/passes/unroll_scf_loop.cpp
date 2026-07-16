//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/permute_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_analysis_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_unroll_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/Affine/Analysis/Utils.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Utils/Utils.h>
#include <mlir/IR/Dominance.h>
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/logger/logger.hpp"

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
    int64_t unrollFactor;
    int64_t depth;
};

//
// UnrollSCFLoopPass
//
class UnrollSCFLoopPass final : public VPU::impl::UnrollSCFLoopBase<UnrollSCFLoopPass> {
public:
    explicit UnrollSCFLoopPass(const UnrollSCFLoopOptions& options, Logger log): UnrollSCFLoopBase(options) {
        Base::initLogger(log, Base::getArgumentName());
    }

    SmallVector<int64_t> parseUnrollFactorsStr(llvm::StringRef unrollFactorStr) {
        SmallVector<int64_t> factors;
        SmallVector<llvm::StringRef, 4> tokens;
        unrollFactorStr.split(tokens, ',');
        for (auto token : tokens) {
            token = token.trim();  // Remove whitespace
            int64_t value = 0;
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
            loops.push_back({forOp.getOperation(), 1, static_cast<int64_t>(depth)});
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
            OpChainAnalysis opChainAnalysis;
            auto opChain = opChainAnalysis.collectParentOpsChain(offsetValue);
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

// Computes an automatic unroll factor from the max trip count of the given scf.for loop.
std::optional<int64_t> computeAutoUnrollFactor(mlir::scf::ForOp forOp, Logger log) {
    OpChainAnalysis opChainAnalysis;
    auto [low, high, step] = opChainAnalysis.getLoopBoundsAndStep(forOp);
    int64_t factor = (high - low) / step;
    if (factor < 2) {
        return std::nullopt;
    }
    log.trace("Auto-unroll: low={0}, high={1}, step={2}, factor={3}", low, high, step, factor);
    return factor;
}

struct LoopTileDimInfo {
    vpux::Dim unrollDim;  // post-permute (NCHW space): indexes into unrollFactor[]
    vpux::Dim tensorDim;  // pre-permute: actual dim in tensor extract/insert ops
    mlir::Value offsetVal;
};

mlir::FailureOr<SmallVector<LoopTileDimInfo>> collectLoopTileDimInfos(mlir::scf::ForOp forOp,
                                                                      mlir::func::FuncOp funcOp) {
    mlir::tensor::ExtractSliceOp firstSliceOp = nullptr;
    mlir::func::FuncOp newFuncOp = nullptr;
    auto directSliceOps = llvm::to_vector(forOp.getOps<mlir::tensor::ExtractSliceOp>());
    auto moduleOp = funcOp->getParentOfType<mlir::ModuleOp>();

    if (!directSliceOps.empty()) {
        firstSliceOp = directSliceOps.front();
    } else {
        auto switchOps = llvm::to_vector(forOp.getOps<mlir::scf::IndexSwitchOp>());
        if (switchOps.empty()) {
            return mlir::failure();
        }
        auto& defaultBlock = switchOps.front().getDefaultBlock();
        auto switchSliceOps = llvm::to_vector(defaultBlock.getOps<mlir::tensor::ExtractSliceOp>());
        if (switchSliceOps.empty()) {
            return mlir::failure();
        }
        firstSliceOp = switchSliceOps.front();
    }

    for (auto user : firstSliceOp.getResult().getUsers()) {
        if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(user)) {
            newFuncOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
        } else if (auto castOp = mlir::dyn_cast<mlir::tensor::CastOp>(user)) {
            for (auto castUser : castOp.getResult().getUsers()) {
                if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(castUser)) {
                    newFuncOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
                    break;
                }
            }
        }

        if (newFuncOp != nullptr) {
            break;
        }
    }

    SmallVector<vpux::Dim> tensorDims;
    SmallVector<mlir::Value> offsetVals;
    for (auto [idx, offset] : llvm::enumerate(firstSliceOp.getMixedOffsets())) {
        auto val = mlir::dyn_cast_or_null<mlir::Value>(offset);
        if (!val) {
            continue;
        }
        auto* defOp = val.getDefiningOp();
        if (defOp != nullptr && mlir::isa<mlir::arith::ConstantOp>(defOp)) {
            continue;
        }
        tensorDims.push_back(vpux::Dim(idx));
        offsetVals.push_back(val);
    }

    if (tensorDims.empty()) {
        return mlir::failure();
    }

    // Use the called function for dim remapping when available so that
    // unrollFactor indices are expressed in the function's output dim space
    // (e.g. NCHW after a PermuteCast). Fall back to the enclosing function
    // when no call is present (inline extract/insert pattern).
    auto funcToUse = (newFuncOp != nullptr) ? newFuncOp : funcOp;
    auto unrollDimsOrFailure = remapDimsThroughInputPermuteCast(funcToUse, tensorDims);
    if (mlir::failed(unrollDimsOrFailure)) {
        return mlir::failure();
    }

    SmallVector<LoopTileDimInfo> result;
    for (auto [tensorDim, unrollDim, offsetVal] : llvm::zip(tensorDims, unrollDimsOrFailure.value(), offsetVals)) {
        result.push_back({unrollDim, tensorDim, offsetVal});
    }

    return result;
}

void mapManualUnrollFactors(mlir::func::FuncOp funcOp, ArrayRef<int64_t> unrollFactor, SmallVector<LoopInfo>& loops,
                            Logger log) {
    funcOp.walk([&](mlir::scf::ForOp forOp) {
        auto infos = collectLoopTileDimInfos(forOp, funcOp);
        if (mlir::failed(infos)) {
            return;
        }
        for (const auto& info : infos.value()) {
            if (checked_cast<size_t>(info.unrollDim.ind()) >= unrollFactor.size() ||
                unrollFactor[info.unrollDim.ind()] <= 1) {
                continue;
            }
            auto parentLoop = findParentForOpFromOffset(info.offsetVal);
            if (parentLoop == nullptr) {
                continue;
            }
            for (auto& loop : loops) {
                if (loop.loopOp != parentLoop.getOperation()) {
                    continue;
                }
                if (loop.unrollFactor > 1 && loop.unrollFactor != unrollFactor[info.unrollDim.ind()]) {
                    log.warning("Loop already assigned factor {0}, overwriting with {1}", loop.unrollFactor,
                                unrollFactor[info.unrollDim.ind()]);
                }
                loop.unrollFactor = unrollFactor[info.unrollDim.ind()];
                break;
            }
        }
    });
}

void mapAutoUnrollFactors(mlir::func::FuncOp funcOp, SmallVector<LoopInfo>& loops, Logger log) {
    llvm::DenseSet<mlir::Operation*> assignedLoops;

    funcOp.walk([&](mlir::scf::ForOp forOp) {
        auto infos = collectLoopTileDimInfos(forOp, funcOp);
        if (mlir::failed(infos)) {
            return;
        }
        for (const auto& info : infos.value()) {
            auto parentLoop = findParentForOpFromOffset(info.offsetVal);
            if (parentLoop == nullptr || assignedLoops.contains(parentLoop.getOperation())) {
                continue;
            }
            auto factor = computeAutoUnrollFactor(parentLoop, log);
            if (!factor.has_value()) {
                continue;
            }
            for (auto& loop : loops) {
                if (loop.loopOp == parentLoop.getOperation()) {
                    loop.unrollFactor = factor.value();
                    log.info("Assigned auto unroll factor {0} to loop with id {1}", factor.value(),
                             parentLoop->getAttrOfType<mlir::IntegerAttr>("id").getInt());
                    assignedLoops.insert(parentLoop.getOperation());
                    break;
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

    for (const auto& allLoopsInBlocks : llvm::make_early_inc_range(loopsToProcess)) {
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

mlir::LogicalResult processUnrolledLoops(mlir::func::FuncOp funcOp, ArrayRef<int64_t> unrollFactor) {
    auto walkResult = funcOp.walk([&](mlir::scf::ForOp forOp) {
        SmallVector<TileDimensionInfo> tileDimensionsInfo;

        auto infos = collectLoopTileDimInfos(forOp, funcOp);
        if (mlir::failed(infos)) {
            return mlir::WalkResult::advance();
        }

        for (const auto& info : infos.value()) {
            auto parentLoop = findParentForOpFromOffset(info.offsetVal);
            if (parentLoop == nullptr) {
                continue;
            }
            auto unrolledFactor = parentLoop->hasAttr("unrolled-factor")
                                          ? parentLoop->getAttrOfType<mlir::IntegerAttr>("unrolled-factor").getInt()
                                          : (checked_cast<size_t>(info.unrollDim.ind()) < unrollFactor.size()
                                                     ? unrollFactor[info.unrollDim.ind()]
                                                     : 1);
            auto dimUnrollFactor = parentLoop->hasAttr("residual") ? 1 : unrolledFactor;
            auto loopIdAttr = parentLoop->getAttrOfType<mlir::IntegerAttr>("id");
            if (!loopIdAttr) {
                continue;
            }
            TileDimensionInfo dimInfo;
            dimInfo.id = loopIdAttr.getInt();
            dimInfo.dimension = info.tensorDim;
            dimInfo.numBlocks = dimUnrollFactor;
            dimInfo.isUnrolled = (dimUnrollFactor > 1);
            dimInfo.forOp = parentLoop;
            tileDimensionsInfo.emplace_back(dimInfo);
        }

        if (tileDimensionsInfo.empty()) {
            return mlir::WalkResult::advance();
        }

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
 * @brief Corrects unrolled loop bounds to ensure memory safety with dynamic tensor dimensions.
 *
 * When MLIR's loopUnrollByFactor calculates loop bounds based on compile-time maximum dimensions,
 * it may generate unsafe bounds that exceed actual runtime tensor sizes.
 * Example: For cascaded unrolling (10→5→2) with dim=1200, initialStep=47:
 *   Loop 1 (×10, step=470): main loop [0,940), epilogue [940,1200)
 *      2xiter
 *   Loop 2 (×5, step=235): main loop [940,1175), epilogue [1175,1200)
 *      1xiter
 *   Loop 3 (×2, step=94): main loop [1175,1269), epilogue [1269,1200) -> out of bounds!
 *   ...
 * This function adds runtime
 * guards to prevent out-of-bounds memory access by dynamically adjusting the main loop's upper bound.
 *
 * The correction strategy:
 * 1. Checks if the first iteration can complete within available runtime data
 * 2. If sufficient data exists: sets upperBound = min(calculatedBound, actualDimSize)
 * 3. If insufficient data: sets upperBound = lowerBound (zero iterations, skip to next loop)
 * 4. Updates epilogue's lowerBound to maintain continuous data coverage
 *
 * Example after postProcessUnrolledLoop: For cascaded unrolling (10→5→2) with dim=1200, initialStep=47:
 *   Loop 1 (×10, step=470): main loop [0,940), epilogue [940,1200)
 *      2xiter
 *   Loop 2 (×5, step=235): main loop [940,1175), epilogue [1175,1200)
 *      1xiter
 *   Loop 3 (×2, step=94): main loop [1175,1175), epilogue [1175,1200) -> no single iteration!
 *      0xiter
 *   Loop 4 (residual loop, ×1, step=47)): [1175,1200)
 *      1xiter (backtrack is applied here if needed)
 *
 * @param result The UnrolledLoopInfo structure from loopUnrollByFactor containing:
 *               - mainLoopOp: The unrolled loop with compile-time calculated bounds
 *               - epilogueLoopOp: The residual loop handling remaining iterations
 * @param builder OpBuilder for creating runtime guard arithmetic operations (AddIOp, CmpIOp, MinUIOp, SelectOp)
 *
 * @return mlir::success() - Bounds successfully corrected and loops updated
 *         mlir::failure() - Should never occur in current implementation (reserved for future validation)
 */
mlir::LogicalResult postProcessUnrolledLoop(mlir::UnrolledLoopInfo& result, mlir::OpBuilder& builder) {
    auto mainLoop = *result.mainLoopOp;
    auto loc = mainLoop.getLoc();
    auto currentLow = mainLoop.getLowerBound();
    auto currentStep = mainLoop.getStep();

    // Get original dimension size from epilogue upper bound - user provided dim size
    auto originalDimSize = result.epilogueLoopOp->getUpperBound();

    // Get DominanceInfo for the parent function
    auto funcOp = mainLoop->getParentOfType<mlir::func::FuncOp>();
    mlir::DominanceInfo dominanceInfo(funcOp);

    // Set insertion point after all dependencies
    auto setInsertionPointAfterValues = [&](mlir::OpBuilder& builder, ArrayRef<mlir::Value> values) {
        mlir::Operation* lastOp = nullptr;
        for (auto val : values) {
            if (auto defOp = val.getDefiningOp()) {
                if (!lastOp || dominanceInfo.dominates(lastOp, defOp)) {
                    lastOp = defOp;
                }
            }
        }
        if (lastOp) {
            builder.setInsertionPointAfter(lastOp);
        }
    };

    SmallVector<mlir::Value> deps = {currentLow, currentStep, originalDimSize, mainLoop.getUpperBound()};
    setInsertionPointAfterValues(builder, deps);

    // Calculate: where will the FIRST iteration end
    auto firstIterationEnd = builder.create<mlir::arith::AddIOp>(loc, currentLow, currentStep);

    // Check: do we have enough user-provided data to ensure the FIRST iteration can ends without out-of-bounds access
    auto canExecute = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ule, firstIterationEnd,
                                                          originalDimSize);
    // Calculate base safe upper bound as min(mainLoop.upperBound, runtime-calculated originalDimSize)
    auto baseSafeUpperbound = builder.create<mlir::arith::MinUIOp>(loc, mainLoop.getUpperBound(), originalDimSize);
    // If there is not enough data to execute even a single iteration, set safe upperbound to current low. It prevents
    // any iterations at all. Execution proceeds to the next loop.
    auto safeUpperbound = builder.create<mlir::arith::SelectOp>(loc, canExecute, baseSafeUpperbound, currentLow);

    // Update bounds. Next for loop starts from the place where current main loop ends
    mainLoop.setUpperBound(safeUpperbound);

    // update epilogue loop lower bound
    result.epilogueLoopOp->setLowerBound(safeUpperbound);
    return mlir::success();
}

/**
 * @brief Unrolls SCF loops by specified factors and sets corresponding attributes on the resulting operations.
 *
 * Processes a collection of loop info structures and unrolls each loop whose factor is > 1.
 * After unrolling, marks the main loop with an "unrolled-factor" attribute and the residual
 * sibling loop (remaining iterations) with a "residual" attribute.
 *
 * In auto-unrolling mode (!hasManualUnrollFactors) two guards are applied per loop based on
 * the autoUnrollingMode:
 *   - Static upper bound: loops with a compile-time constant upper bound are skipped (all modes).
 *   - Nesting guard (mode-dependent):
 *       INNER – skip outer loop if any direct descendant ForOp already carries factor > 1
 *       OUTER – skip inner loop if any ancestor loop in the candidate set has factor > 1
 *       ALL   – no nesting guard; all loops with factor > 1 are unrolled.
 *       BIGGEST – within each nesting chain (ancestors + descendants of a loop), unroll only
 *               the loop with the maximum factor; sibling loops (no ancestor-descendant
 *               relationship) are compared independently.
 *
 * @param builder              MLIR OpBuilder for IR mutations.
 * @param loops                Loop info structures (loop op + unroll factor).
 * @param enableCascadedUnrolling  Apply cascaded stages N → N/2 → N/4 → … instead of a single pass.
 * @param hasManualUnrollFactors   True when factors come from loop-unroll-factor; disables auto-mode guards.
 * @param autoUnrollingMode    Controls which loop nesting levels are eligible for auto-unrolling.
 * @param log                  Logger.
 *
 * @return mlir::success() if all unrolling operations complete; mlir::failure() otherwise.
 */
mlir::LogicalResult unrollLoopsAndSetAttributes(mlir::OpBuilder& builder, SmallVector<LoopInfo>& loops,
                                                bool enableCascadedUnrolling, bool hasManualUnrollFactors,
                                                AutoUnrollingMode autoUnrollingMode, Logger log) {
    // Returns the subset of candidate loops related to currentOp under the given mode.
    //
    //   INNER   → descendants only  (inner / child loops of currentOp)
    //   OUTER   → ancestors only    (outer / parent loops of currentOp)
    //   ALL     → nesting chain     (ancestors + descendants; siblings excluded)
    //   BIGGEST → nesting chain including currentOp itself, then filtered to the single loop
    //             with the maximum unrollFactor. On a tie, the first entry in the candidate
    //             set wins. Exactly one winner is returned to prevent 2-D unrolling.
    //             The returned set is the "winners": if currentOp is in it, unroll it;
    //             if not, skip it.
    //
    // In all cases loops with factor <= 1 are excluded.
    // currentOp itself is excluded for INNER / OUTER / ALL, but included for BIGGEST.
    auto getCompetingLoops = [&loops](mlir::Operation* currentOp,
                                      AutoUnrollingMode mode) -> SmallVector<const LoopInfo*> {
        SmallVector<const LoopInfo*> result;
        for (const auto& candidate : loops) {
            if (candidate.loopOp == currentOp || candidate.unrollFactor <= 1) {
                continue;
            }
            const bool isAncestor = candidate.loopOp->isProperAncestor(currentOp);
            const bool isDescendant = currentOp->isProperAncestor(candidate.loopOp);
            switch (mode) {
            case AutoUnrollingMode::INNER:
                if (!isDescendant) {
                    continue;
                }
                break;
            case AutoUnrollingMode::OUTER:
                if (!isAncestor) {
                    continue;
                }
                break;
            case AutoUnrollingMode::ALL:
            case AutoUnrollingMode::BIGGEST:
                if (!isAncestor && !isDescendant) {
                    continue;
                }
                break;
            default:
                continue;
            }
            result.push_back(&candidate);
        }
        if (mode == AutoUnrollingMode::BIGGEST) {
            // Add currentOp itself so the winner set is self-contained.
            for (const auto& candidate : loops) {
                if (candidate.loopOp == currentOp && candidate.unrollFactor > 1) {
                    result.push_back(&candidate);
                    break;
                }
            }
            int64_t maxFactor = 0;
            for (const auto* info : result) {
                maxFactor = std::max(maxFactor, info->unrollFactor);
            }
            llvm::erase_if(result, [maxFactor](const LoopInfo* info) {
                return info->unrollFactor < maxFactor;
            });
            // Multiple loops with equal max factor would cause 2-D unrolling — keep only one.
            // collectForOpsInnerToOuter fills `loops` innermost-first, so the surviving loop
            // is the innermost one (inner loop wins over outer loop in a 2-D tiling case and same automatically
            // assigned unroll factor).
            if (result.size() > 1) {
                result.resize(1);
            }
        }
        return result;
    };

    for (auto& loopInfo : loops) {
        if (loopInfo.unrollFactor == 1) {
            continue;
        }

        auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(loopInfo.loopOp);
        auto loopId = forOp->getAttrOfType<mlir::IntegerAttr>("id");

        if (!hasManualUnrollFactors) {
            // Guard 1 (all modes): skip loops with a static (compile-time constant) upper bound.
            // When the trip count is evenly divisible by the unroll factor, unrolling produces
            // no epilogue and the main loop collapses to a single iteration, which is then
            // canonicalized and inlined — eliminating the scf.for entirely and breaking
            // downstream passes that rely on the loop structure. Conservative guard: skip all
            // static loops.
            // Track: E#218830
            if (mlir::getConstantIntValue(forOp.getUpperBound()).has_value()) {
                log.info("Skipping auto-unrolling for loop with id {0} because upper bound is static", loopId.getInt());
                continue;
            }
            // Guard 2: nesting guard — behaviour depends on autoUnrollingMode.
            if (autoUnrollingMode == AutoUnrollingMode::INNER) {
                // INNER: skip an outer loop when any descendant loop in the candidate set has unrollFactor > 1
                if (!getCompetingLoops(forOp, AutoUnrollingMode::INNER).empty()) {
                    log.info("Skipping loop {0} (INNER mode: descendant loop has unroll factor > 1)", loopId.getInt());
                    continue;
                }
            } else if (autoUnrollingMode == AutoUnrollingMode::OUTER) {
                // OUTER: skip an inner loop when any ancestor loop in the candidate set already has unrollFactor > 1
                if (!getCompetingLoops(forOp, AutoUnrollingMode::OUTER).empty()) {
                    log.info("Skipping loop {0} (OUTER mode: ancestor loop has unroll factor > 1)", loopId.getInt());
                    continue;
                }
            } else if (autoUnrollingMode == AutoUnrollingMode::BIGGEST) {
                // BIGGEST: getCompetingLoops returns exactly one winner — the loop with the
                // largest factor in the nesting chain. Skip the current loop if it is not that winner.
                const auto winners = getCompetingLoops(forOp, AutoUnrollingMode::BIGGEST);
                if (winners.size() != 1) {
                    log.error("BIGGEST mode must produce exactly one winner, got {0}", winners.size());
                    return mlir::failure();
                }
                if (winners.front()->loopOp != forOp.getOperation()) {
                    log.info("Skipping loop {0} (BIGGEST mode: factor {1} < max factor {2} in nesting chain)",
                             loopId.getInt(), loopInfo.unrollFactor, winners.front()->unrollFactor);
                    continue;
                }
            }
        }
        // AutoUnrollingMode::ALL: no nesting guard — all loops with factor > 1 are unrolled.

        log.info("Unrolling loop with id {0} by factor {1}", loopId.getInt(), loopInfo.unrollFactor);

        int64_t currentFactor = loopInfo.unrollFactor;
        SmallVector<int64_t> unrollFactors{currentFactor};

        if (enableCascadedUnrolling) {
            log.trace("Cascaded unrolling enabled for loop id {0}", loopId.getInt());
            // Cascaded unrolling: 10 -> 5 -> 2 -> epilogue
            // Generate sequence of factors by dividing by 2
            while ((currentFactor / 2) > 1) {
                currentFactor = currentFactor / 2;
                unrollFactors.push_back(currentFactor);
            }
        }

        mlir::scf::ForOp currentLoop = forOp;

        // Apply unrolling with each factor in sequence
        for (size_t i = 0; i < unrollFactors.size(); ++i) {
            int64_t factor = unrollFactors[i];
            log.trace("Applying unroll factor {0} for loop id {1}", factor, loopId.getInt());

            auto result = mlir::loopUnrollByFactor(currentLoop, factor);

            if (mlir::failed(result)) {
                return mlir::failure();
            }

            // Mark the main loop with unroll attribute
            if (result->mainLoopOp.has_value()) {
                result->mainLoopOp->getOperation()->setAttr("unrolled-factor", builder.getI64IntegerAttr(factor));
                result->mainLoopOp->getOperation()->setAttr("id", loopId);
                moveAffineArithOpsEarly(*result->mainLoopOp->getBody());
            }

            // change loop id to avoid id's duplication
            // new id = int(str(loopId.getInt()) + str(current_unroll_factor))
            auto loopIdStr = std::to_string(loopId.getInt()) + std::to_string(unrollFactors[i]);
            int64_t newLoopIdInt = std::stoll(loopIdStr);
            currentLoop->setAttr(
                    "id", mlir::IntegerAttr::get(mlir::IntegerType::get(currentLoop.getContext(), 64), newLoopIdInt));

            if (!result->epilogueLoopOp.has_value()) {
                // static dims to unroll and no epilogue left
                break;
            }
            // The following attributes will be referred in ConvertToLLVMUMD pass
            result->mainLoopOp->getOperation()->setAttr("no_await_all", builder.getBoolAttr(true));

            if (i > 0) {
                // No need of resetting command list for cascaded unrolling except for the first unroll
                // as all cascaded unrolls are part of single command list
                // The following attributes will be referred in ConvertToLLVMUMD pass
                result->mainLoopOp->getOperation()->setAttr("no_reset_cmdlist", builder.getBoolAttr(true));
            }

            // Post-process main unrolled loop to fix bounds and add runtime guards
            if (mlir::failed(postProcessUnrolledLoop(*result, builder))) {
                log.error("Failed to post-process unrolled loop with id {0}", loopId.getInt());
                return mlir::failure();
            }

            // Process epilogue - it becomes the next loop to unroll
            result->epilogueLoopOp->getOperation()->setAttr("id", loopId);

            // If this is the last unroll factor, mark epilogue as residual
            if (i == unrollFactors.size() - 1) {
                result->epilogueLoopOp->getOperation()->setAttr("residual", builder.getUnitAttr());
                // The following attributes will be referred in ConvertToLLVMUMD pass
                result->epilogueLoopOp->getOperation()->setAttr("no_reset_cmdlist", builder.getBoolAttr(true));
            }

            // residual will be unrolled with smaller unroll factor in the next iteration
            currentLoop = *result->epilogueLoopOp;

            moveAffineArithOpsEarly(*result->mainLoopOp->getBody());
        }
    }

    return mlir::success();
}

void cleanupAttributes(mlir::func::FuncOp funcOp) {
    funcOp.walk([&](mlir::scf::ForOp forOp) {
        if (forOp->hasAttr("id")) {
            forOp->removeAttr("id");
        }

        if (forOp->hasAttr("unrolled-factor")) {
            forOp->removeAttr("unrolled-factor");
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
    auto mainFuncOp = net::getMainFunc(moduleOp);
    mlir::OpBuilder builder(mainFuncOp);

    const auto autoUnrollingModeValue = autoUnrollingMode.getValue();
    if (_loopUnrollFactor.empty() && autoUnrollingModeValue == AutoUnrollingMode::DISABLED) {
        _log.trace("Unroll factor not specified and auto-unrolling disabled. Skipping unroll scf pass");
        return;
    }

    const bool hasManualFactors = !_loopUnrollFactor.empty();

    SmallVector<int64_t> initialUnrollFactors;
    if (hasManualFactors) {
        initialUnrollFactors = parseUnrollFactorsStr(_loopUnrollFactor);
        if (initialUnrollFactors.empty()) {
            _log.error("Failed to parse unroll factors. Skipping unroll scf pass");
            signalPassFailure();
            return;
        }
    }

    // Early validation: skip if all manual factors are <= 1
    if (hasManualFactors) {
        auto numDimsToUnroll = llvm::count_if(initialUnrollFactors, [](auto factor) {
            return factor > 1;
        });

        if (numDimsToUnroll == 0) {
            _log.trace("No unroll factors greater than 1 specified. Skipping unroll scf pass");
            return;
        }
    }

    /* Collect all loops in the function from innermost to outermost.
       This ensures innermost loops are unrolled first; outer loops that also
       need unrolling are then processed after. */
    SmallVector<LoopInfo> loopInfoVector;
    collectForOpsInnerToOuter(mainFuncOp.getOperation(), loopInfoVector);
    if (loopInfoVector.empty()) {
        _log.trace("No loops found. Skipping unroll scf pass");
        return;
    }

    // Manual factors take priority over auto-unrolling. The two modes are mutually exclusive.
    if (hasManualFactors) {
        mapManualUnrollFactors(mainFuncOp, initialUnrollFactors, loopInfoVector, _log);
    } else {
        mapAutoUnrollFactors(mainFuncOp, loopInfoVector, _log);
    }

    auto numLoopsToUnroll = llvm::count_if(loopInfoVector, [](const LoopInfo& info) {
        return info.unrollFactor > 1;
    });
    if (numLoopsToUnroll == 0) {
        _log.trace("No loops with unroll factor > 1 found after mapping. Skipping unroll scf pass");
        return;
    }

    if (mlir::failed(unrollLoopsAndSetAttributes(builder, loopInfoVector, enableCascadedUnrolling.getValue(),
                                                 hasManualFactors, autoUnrollingModeValue, _log))) {
        signalPassFailure();
        return;
    }

    if (mlir::failed(fuseSiblingUnrolledLoops(mainFuncOp))) {
        _log.trace("Failed to fuse sibling unrolled loops");
        signalPassFailure();
        return;
    }

    if (mlir::failed(processUnrolledLoops(mainFuncOp, initialUnrollFactors))) {
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

std::unique_ptr<mlir::Pass> vpux::VPU::createUnrollSCFLoopPass(StringRef loopUnrollFactor, bool enableCascadedUnrolling,
                                                               AutoUnrollingMode autoUnrollingMode, Logger log) {
    UnrollSCFLoopOptions options;
    options.loopUnrollFactor = loopUnrollFactor.str();
    options.enableCascadedUnrolling = enableCascadedUnrolling;
    options.autoUnrollingMode = autoUnrollingMode;
    return std::make_unique<UnrollSCFLoopPass>(options, log);
}
