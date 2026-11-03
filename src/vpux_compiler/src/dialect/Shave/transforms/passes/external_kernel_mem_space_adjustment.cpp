//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <numeric>
#include <optional>

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"
#include "vpux/compiler/dialect/Shave/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/hw_settings.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux::Shave {
#define GEN_PASS_DECL_EXTERNALKERNELMEMSPACEADJUSTMENT
#define GEN_PASS_DEF_EXTERNALKERNELMEMSPACEADJUSTMENT
#include "vpux/compiler/dialect/Shave/passes.hpp.inc"
}  // namespace vpux::Shave

using namespace vpux;

namespace {

// ─── Shared helpers ──────────────────────────────────────────────────────────

// Total aligned CMX bytes for a set of values.  Returns 0 for an empty set.
static int64_t computeAlignedCMXSize(mlir::ArrayRef<mlir::Value> vals) {
    if (vals.empty()) {
        return 0;
    }
    SmallVector<Byte> sizes;
    sizes.reserve(vals.size());
    for (auto val : vals) {
        sizes.push_back(mlir::cast<vpux::NDTypeInterface>(val.getType()).getTotalAllocSize());
    }
    return vpux::calculateAlignedBuffersMemoryRequirement(sizes, Byte(vpux::DEFAULT_CMX_ALIGNMENT), Byte(1)).count();
}

// Returns the index-subset of vals that fits within budget bytes,
// maximising CMX utilisation (same knapsack heuristic as AdjustMemorySpaceForSHVOps).
static mlir::DenseSet<size_t> fitSubsetInCMX(mlir::ArrayRef<mlir::Value> vals, int64_t budget) {
    mlir::DenseSet<size_t> result;
    if (vals.empty() || budget <= 0) {
        return result;
    }

    SmallVector<Byte> sizes;
    sizes.reserve(vals.size());
    for (auto val : vals) {
        sizes.push_back(mlir::cast<vpux::NDTypeInterface>(val.getType()).getTotalAllocSize());
    }

    const Byte offsetAlign{vpux::DEFAULT_CMX_ALIGNMENT};
    const Byte sizeAlign{1};

    // Fast path: all values fit.
    if (vpux::calculateAlignedBuffersMemoryRequirement(sizes, offsetAlign, sizeAlign).count() <= budget) {
        for (size_t i = 0; i < vals.size(); ++i) {
            result.insert(i);
        }
        return result;
    }

    SmallVector<size_t> idxVec(vals.size());
    std::iota(idxVec.begin(), idxVec.end(), 0);

    SmallVector<std::pair<SmallVector<size_t>, int64_t>> subsets;
    SmallVector<size_t> aux;
    auto genSubsets = [](const auto& iv, auto& ss, auto& a, size_t cur, auto& self) -> void {
        ss.push_back({a, 0});
        for (size_t i = cur; i < iv.size(); ++i) {
            a.push_back(iv[i]);
            self(iv, ss, a, i + 1, self);
            a.pop_back();
        }
    };
    genSubsets(idxVec, subsets, aux, 0, genSubsets);

    for (auto& p : subsets) {
        SmallVector<Byte> sub;
        for (auto idx : p.first) {
            sub.push_back(sizes[idx]);
        }
        p.second =
                sub.empty() ? 0 : vpux::calculateAlignedBuffersMemoryRequirement(sub, offsetAlign, sizeAlign).count();
    }
    std::sort(subsets.begin(), subsets.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    std::optional<size_t> best;
    for (size_t i = 0; i < subsets.size(); ++i) {
        if (subsets[i].second > budget) {
            if (i > 0) {
                best = i - 1;
            }
            break;
        }
    }
    if (!best) {
        return result;
    }
    for (auto idx : subsets[*best].first) {
        result.insert(idx);
    }
    return result;
}

// Returns the configured ScratchLocation for an ExternalKernelOp scratch operand,
// or throws when the operand is not a scratch buffer
static Shave::ScratchLocation getScratchLocationForOperand(Shave::ExternalKernelOp op, unsigned operandIdx) {
    auto locAttr = op.getScratchLocationsAttr();
    VPUX_THROW_UNLESS(locAttr, "Scratch locations need to be explicitly set");

    const auto numInputs = static_cast<unsigned>(op.getInputs().size());
    const auto scratchIdx = operandIdx - numInputs;
    const auto locations = locAttr.getValue();
    VPUX_THROW_WHEN(operandIdx < numInputs || scratchIdx >= locations.size(),
                    "Scratch location can only be retrieved for valid scratch buffers (idx={0}, numInputs={1})",
                    operandIdx, numInputs);

    return mlir::cast<Shave::ScratchLocationAttr>(locations[scratchIdx]).getValue();
}

// Returns true when any operand or result of op carries a DistributedTensorType.
// Such ops are handled by the distributed bufferization path and must be skipped.
static bool hasDistributedType(mlir::Operation* op) {
    auto isDistributed = [](mlir::Value value) {
        if (auto distributedIf = mlir::dyn_cast<vpux::VPU::DistributedTypeInterface>(value.getType())) {
            return distributedIf.containsDistributedTypes();
        }
        return false;
    };
    return llvm::any_of(op->getOperands(), isDistributed) || llvm::any_of(op->getResults(), isDistributed);
}

// ─── ExternalKernelOp placement ──────────────────────────────────────────────
//
// Staged CMX-placement for ExternalKernelOp scratch and IO buffers.
//
// Stage 1 – CMX-pinned scratch (mandatory): all such buffers MUST fit; fatal error otherwise.
// Stage 2 – DDR-pinned scratch: left in DDR, budget untouched.
// Stage 3 – IO inputs (skipped for uses_dma kernels): placed greedily within remaining budget.
// Stage 4 – outputs (skipped for uses_dma kernels): placed greedily within remaining budget.
// Stage 5 – AUTO scratch (best-effort, lowest priority): placed in remaining CMX if possible; DDR otherwise.

static mlir::LogicalResult moveExternalKernelIntoCMX(Shave::ExternalKernelOp extKernelOp, IndexedSymbolAttr memSpaceCMX,
                                                     int64_t totalAvailableCMX, Logger& log) {
    mlir::Operation* op = extKernelOp.getOperation();
    mlir::OpBuilder builder(op);
    int64_t remainingCMX = totalAvailableCMX;

    log.trace("Moving ExternalKernelOp '{0}' into CMX (total budget: {1} bytes)", op->getLoc(), totalAvailableCMX);

    // Partition scratch buffers by placement policy.
    SmallVector<mlir::Value> cmxScratch, ddrScratch, autoScratch;
    for (auto& operand : extKernelOp.getScratchBuffersMutable()) {
        auto loc = getScratchLocationForOperand(extKernelOp, operand.getOperandNumber());
        if (loc == Shave::ScratchLocation::CMX) {
            cmxScratch.push_back(operand.get());
        } else if (loc == Shave::ScratchLocation::DDR) {
            ddrScratch.push_back(operand.get());
        } else if (loc == Shave::ScratchLocation::AUTO) {
            autoScratch.push_back(operand.get());
        } else {
            VPUX_THROW("Unknown scratch location for operand {0} of ExternalKernelOp at '{1}'",
                       operand.getOperandNumber(), op->getLoc());
        }
    }

    // ── Stage 1: CMX-pinned scratch ──────────────────────────────────────────
    if (!cmxScratch.empty()) {
        auto fitSet = fitSubsetInCMX(cmxScratch, remainingCMX);
        if (fitSet.size() != cmxScratch.size()) {
            log.error("ExternalKernelOp at '{0}': CMX-pinned scratch buffers require {1} bytes but only "
                      "{2} bytes are available. Compilation cannot continue.",
                      op->getLoc(), computeAlignedCMXSize(cmxScratch), remainingCMX);
            // Signal through the caller.
            return mlir::failure();
        }
        for (auto val : cmxScratch) {
            auto emptyOp = val.getDefiningOp<VPU::EmptyOp>();
            VPUX_THROW_UNLESS(emptyOp != nullptr, "CMX-pinned scratch buffer must be backed by VPU::EmptyOp");
            emptyOp.getResult().setType(mlir::cast<mlir::RankedTensorType>(
                    mlir::cast<NDTypeInterface>(emptyOp.getType()).changeMemSpace(memSpaceCMX)));
        }
        remainingCMX -= computeAlignedCMXSize(cmxScratch);
        log.nest().trace("Stage 1: placed {0} CMX-pinned scratch buffer(s); {1} bytes remaining", cmxScratch.size(),
                         remainingCMX);
    }

    // ── Stage 2: DDR-pinned scratch ──────────────────────────────────────────
    log.nest().trace("Stage 2: {0} DDR-pinned scratch buffer(s) remain in DDR", ddrScratch.size());

    // ── Stage 3: IO inputs (skip for uses_dma) ───────────────────────────────
    const bool usesDma = extKernelOp.getUsesDma();
    if (!usesDma) {
        SmallVector<mlir::Value> inputVals(extKernelOp.getInputs().begin(), extKernelOp.getInputs().end());
        auto fitSet = fitSubsetInCMX(inputVals, remainingCMX);
        SmallVector<mlir::Value> placedInputs;
        for (size_t i = 0; i < inputVals.size(); ++i) {
            if (!fitSet.count(i)) {
                continue;
            }
            auto& operand = op->getOpOperand(static_cast<unsigned>(i));
            auto origType = mlir::cast<vpux::NDTypeInterface>(operand.get().getType());
            placedInputs.push_back(operand.get());
            if (origType.getMemSpace() != memSpaceCMX) {
                auto copied = builder.createOrFold<VPU::CopyOp>(appendLoc(op->getLoc(), "input-{0}-CMX", i),
                                                                operand.get(), memSpaceCMX);
                operand.set(copied);
            }
        }
        remainingCMX -= computeAlignedCMXSize(placedInputs);
        log.nest().trace("Stage 3: placed {0}/{1} input(s) in CMX; {2} bytes remaining", fitSet.size(),
                         inputVals.size(), remainingCMX);
    } else {
        log.nest().trace("Stage 3: skipped (uses_dma kernel; inputs handled by DMA engine)");
    }

    // ── Stage 4: outputs (skip for uses_dma) ─────────────────────────────
    if (!usesDma) {
        SmallVector<mlir::Value> outputVals;
        for (auto result : op->getOpResults()) {
            outputVals.push_back(result);
        }
        auto outputFitSet = fitSubsetInCMX(outputVals, remainingCMX);
        SmallVector<mlir::Value> placedOutputs;
        for (auto result : llvm::make_early_inc_range(op->getOpResults())) {
            if (!outputFitSet.count(result.getResultNumber())) {
                continue;
            }
            auto origType = mlir::cast<vpux::NDTypeInterface>(result.getType());
            if (origType.getMemSpace() == memSpaceCMX) {
                placedOutputs.push_back(result);
                continue;
            }
            builder.setInsertionPointAfter(op);
            result.setType(origType.changeMemSpace(memSpaceCMX));
            auto copiedOut = builder.createOrFold<VPU::CopyOp>(appendLoc(op->getLoc(), "output-DDR"), result,
                                                               origType.getMemSpace());
            result.replaceAllUsesExcept(copiedOut, copiedOut.getDefiningOp());
            placedOutputs.push_back(result);
        }
        remainingCMX -= computeAlignedCMXSize(placedOutputs);
        log.nest().trace("Stage 4: placed {0}/{1} output(s) in CMX; {2} bytes remaining", placedOutputs.size(),
                         outputVals.size(), remainingCMX);
    } else {
        log.nest().trace("Stage 4: skipped (uses_dma kernel; outputs handled by DMA engine)");
    }

    // ── Stage 5: AUTO scratch (best-effort) ──────────────────────────────────
    if (!autoScratch.empty()) {
        auto fitSet = fitSubsetInCMX(autoScratch, remainingCMX);
        SmallVector<mlir::Value> placedAuto;
        for (size_t i = 0; i < autoScratch.size(); ++i) {
            auto val = autoScratch[i];
            if (fitSet.count(i)) {
                auto emptyOp = val.getDefiningOp<VPU::EmptyOp>();
                VPUX_THROW_UNLESS(emptyOp != nullptr, "AUTO scratch buffer must be backed by VPU::EmptyOp");
                emptyOp.getResult().setType(mlir::cast<mlir::RankedTensorType>(
                        mlir::cast<NDTypeInterface>(emptyOp.getType()).changeMemSpace(memSpaceCMX)));
                placedAuto.push_back(val);
                log.info("ExternalKernelOp '{0}': AUTO scratch buffer {1} placed in CMX", op->getLoc(), i);
            } else {
                log.info("ExternalKernelOp '{0}': AUTO scratch buffer {1} remains in DDR (insufficient CMX budget)",
                         op->getLoc(), i);
            }
        }
        remainingCMX -= computeAlignedCMXSize(placedAuto);
    }

    return mlir::success();
}

// ─── RepeatedExternalKernelOp placement ──────────────────────────────────────
//
// Verify placement for REPEATED kernels (IO is never in CMX, CMX scratch buffers shall be DistributedTensors, aka
// already in CMX).
//
// Stage 1 – cmx_scratch_buffers: mandatory CMX placement; fatal if budget exceeded.
// Stage 2 – ddr_scratch_buffers: left in DDR unchanged.

static mlir::LogicalResult verifyRepeatedExternalKernelBufferPlacement(Shave::RepeatedExternalKernelOp repeatedOp,
                                                                       IndexedSymbolAttr memSpaceCMX,
                                                                       int64_t totalAvailableCMX, Logger& log) {
    log.trace("Verifying RepeatedExternalKernelOp '{0}' placement; with total CMX budget: {1} bytes",
              repeatedOp->getLoc(), totalAvailableCMX);

    SmallVector<mlir::Value> cmxScratch(repeatedOp.getCmxScratchBuffers().begin(),
                                        repeatedOp.getCmxScratchBuffers().end());

    // ── Stage 1: cmx_scratch_buffers (mandatory) ─────────────────────────────
    if (!cmxScratch.empty()) {
        const int64_t required = computeAlignedCMXSize(cmxScratch);
        if (required > totalAvailableCMX) {
            log.error("RepeatedExternalKernelOp at '{0}': cmx_scratch_buffers require {1} bytes but only "
                      "{2} bytes are available. Compilation cannot continue.",
                      repeatedOp->getLoc(), required, totalAvailableCMX);
            return mlir::failure();
        }

        for (auto val : cmxScratch) {
            if (mlir::isa<VPU::DistributedTypeInterface>(val.getType())) {
                // Already distributed, assumed to be correctly placed in CMX; skip.
                continue;
            } else {
                log.warning(
                        "RepeatedExternalKernelOp at '{0}': cmx_scratch_buffer '{1}' is not a DistributedTensorType; "
                        "this leads to the CMX scratch buffer being uniquely placed in CMX but not distributed across "
                        "tiles. This may lead to errors.",
                        repeatedOp->getLoc(), val);
                auto emptyOp = val.getDefiningOp<VPU::EmptyOp>();
                VPUX_THROW_UNLESS(emptyOp != nullptr, "CMX-pinned scratch buffer must be backed by VPU::EmptyOp");
                emptyOp.getResult().setType(mlir::cast<mlir::RankedTensorType>(
                        mlir::cast<NDTypeInterface>(emptyOp.getType()).changeMemSpace(memSpaceCMX)));
            }
        }

        log.nest().trace("RepeatedExternalKernelOp: {0} cmx_scratch_buffer(s) correctly placed/distributed in CMX",
                         cmxScratch.size());
    }

    // ── Stage 2: ddr_scratch_buffers (no-op) ─────────────────────────────────
    log.nest().trace("RepeatedExternalKernelOp: {0} ddr_scratch_buffer(s) remain in DDR",
                     repeatedOp.getDdrScratchBuffers().size());
    // IO inputs and outputs are always skipped for REPEATED kernels (uses_dma).

    return mlir::success();
}

// ─── Pass ─────────────────────────────────────────────────────────────────────

class ExternalKernelMemSpaceAdjustment final :
        public Shave::impl::ExternalKernelMemSpaceAdjustmentBase<ExternalKernelMemSpaceAdjustment> {
public:
    explicit ExternalKernelMemSpaceAdjustment(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ExternalKernelMemSpaceAdjustment::safeRunOnFunc() {
    auto func = getOperation();
    auto moduleOp = func->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_UNLESS(moduleOp, "ExternalKernelMemSpaceAdjustment: could not find parent ModuleOp");

    const auto totalAvailableCMX = VPU::getTotalCMXSize(moduleOp).count();
    const auto memSpaceCMX = IndexedSymbolAttr::get(&getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN), 0);

    // Collect ops before iterating to avoid invalidation during placement mutations.
    SmallVector<Shave::ExternalKernelOp> externalKernelOps;
    SmallVector<Shave::RepeatedExternalKernelOp> repeatedKernelOps;
    func.walk([&](mlir::Operation* op) {
        if (auto extOp = mlir::dyn_cast<Shave::ExternalKernelOp>(op)) {
            externalKernelOps.push_back(extOp);
        } else if (auto repOp = mlir::dyn_cast<Shave::RepeatedExternalKernelOp>(op)) {
            repeatedKernelOps.push_back(repOp);
        }
    });

    for (auto extKernelOp : externalKernelOps) {
        // Skip ops whose operands or results already carry DistributedTensorType;
        // those are handled by the distributed bufferization path.
        VPUX_THROW_WHEN(hasDistributedType(extKernelOp.getOperation()),
                        "ExternalKernelOp at '{0}' has distributed operand or result type; this is unexpected",
                        extKernelOp.getLoc());
        if (mlir::failed(moveExternalKernelIntoCMX(extKernelOp, memSpaceCMX, totalAvailableCMX, _log))) {
            signalPassFailure();
            return;
        }
    }

    for (auto repeatedOp : repeatedKernelOps) {
        auto res = verifyRepeatedExternalKernelBufferPlacement(repeatedOp, memSpaceCMX, totalAvailableCMX, _log);
        if (mlir::failed(res)) {
            signalPassFailure();
            return;
        }
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::Shave::createExternalKernelMemSpaceAdjustmentPass(Logger log) {
    return std::make_unique<ExternalKernelMemSpaceAdjustment>(std::move(log));
}
