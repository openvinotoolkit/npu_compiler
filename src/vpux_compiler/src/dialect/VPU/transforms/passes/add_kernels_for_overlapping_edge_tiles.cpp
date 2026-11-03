//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/params.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_analysis_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/MathExtras.h>
#include <mlir/Analysis/SliceAnalysis.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_ADDKERNELSFOROVERLAPPINGEDGETILES
#define GEN_PASS_DEF_ADDKERNELSFOROVERLAPPINGEDGETILES
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

//
// Low resolution range of a dynamic dim, with the notation used in this file:
//   S    - loop step (size of one output tile)
//   D    - runtime output dim; the low resolution range is S < D <= S + halo
//   bt   - offset of the output tile (after backtracking)
//   O_l  - number of invalid outputs at the start of a tile computed with full left pad (`invalid_output_left`)
// AdjustBlockSizeForScfTiling computes D in this range with two FULLBLK tiles with full pads: tile 0 at bt = 0 and
// tile 1 at bt = D - S. The first O_l outputs of tile 1 read the pad instead of real input; they are taken from the
// accumulator, where tile 0 wrote valid values
//

namespace {

// Returns true if `value` is computed from the induction variable of `forOp` by ops of the loop body
bool dependsOnInductionVar(mlir::Value value, mlir::scf::ForOp forOp) {
    const auto inductionVar = forOp.getInductionVar();
    if (value == inductionVar) {
        return true;
    }
    mlir::ForwardSliceOptions sliceOptions;
    sliceOptions.filter = [&](mlir::Operation* op) {
        return forOp->isProperAncestor(op);
    };
    llvm::SetVector<mlir::Operation*> forwardSlice;
    mlir::getForwardSlice(inductionVar, &forwardSlice, sliceOptions);
    return forwardSlice.contains(value.getDefiningOp());
}

// The switch closest to `insertOp` among the ops of the body of `forOp` its source is computed from
mlir::scf::IndexSwitchOp getIndexSwitchOp(mlir::tensor::InsertSliceOp insertOp, mlir::scf::ForOp forOp) {
    mlir::BackwardSliceOptions sliceOptions;
    sliceOptions.omitBlockArguments = true;
    sliceOptions.inclusive = true;
    sliceOptions.filter = [&](mlir::Operation* op) {
        return forOp->isProperAncestor(op);
    };
    llvm::SetVector<mlir::Operation*> backwardSlice;
    if (mlir::failed(mlir::getBackwardSlice(insertOp.getSource(), &backwardSlice, sliceOptions))) {
        return nullptr;
    }
    // The slice is topologically sorted: the last switch is the closest to the insert
    auto switchIt = llvm::find_if(llvm::reverse(backwardSlice), [](mlir::Operation* op) {
        return mlir::isa<mlir::scf::IndexSwitchOp>(op);
    });
    return switchIt != backwardSlice.rend() ? mlir::cast<mlir::scf::IndexSwitchOp>(*switchIt) : nullptr;
}

std::string optionalToString(std::optional<int64_t> value) {
    return value.has_value() ? std::to_string(value.value()) : "unknown";
}

bool hasLowResSliceId(mlir::tensor::ExtractSliceOp sliceOp, int64_t id) {
    auto ids = sliceOp->getAttrOfType<mlir::DenseI64ArrayAttr>(VPU::LOW_RES_SLICE_ATTR_NAME);
    return ids != nullptr && llvm::is_contained(ids.asArrayRef(), id);
}

// Removes `id` from the tag of `sliceOp`, and the tag once it is empty
void removeLowResSliceId(mlir::tensor::ExtractSliceOp sliceOp, int64_t id) {
    auto ids = sliceOp->getAttrOfType<mlir::DenseI64ArrayAttr>(VPU::LOW_RES_SLICE_ATTR_NAME);
    auto remaining = llvm::to_vector(llvm::make_filter_range(ids.asArrayRef(), [&](int64_t other) {
        return other != id;
    }));
    if (remaining.empty()) {
        sliceOp->removeAttr(VPU::LOW_RES_SLICE_ATTR_NAME);
    } else {
        sliceOp->setAttr(VPU::LOW_RES_SLICE_ATTR_NAME, mlir::DenseI64ArrayAttr::get(sliceOp.getContext(), remaining));
    }
}

// Replaces the offset of `sliceOp` on `dim` by `offset`, keeping its other attributes
mlir::tensor::ExtractSliceOp setSliceOffset(mlir::tensor::ExtractSliceOp sliceOp, int64_t dim, mlir::Value offset) {
    auto offsets = sliceOp.getMixedOffsets();
    offsets[dim] = offset;
    mlir::OpBuilder builder(sliceOp);
    auto newSliceOp =
            mlir::tensor::ExtractSliceOp::create(builder, sliceOp.getLoc(), sliceOp.getType(), sliceOp.getSource(),
                                                 offsets, sliceOp.getMixedSizes(), sliceOp.getMixedStrides());
    newSliceOp->setDiscardableAttrs(sliceOp->getDiscardableAttrDictionary());
    sliceOp.replaceAllUsesWith(newSliceOp.getResult());
    sliceOp.erase();
    return newSliceOp;
}

// The call that produces the tile of a case region: the region must have a single call and yield its single result
mlir::func::CallOp getCallOpFromCaseRegion(mlir::Region& caseRegion) {
    SmallVector<mlir::func::CallOp> callOps;
    caseRegion.front().walk([&](mlir::func::CallOp callOp) {
        callOps.push_back(callOp);
    });
    if (callOps.size() != 1 || callOps.front().getNumResults() != 1) {
        return nullptr;
    }
    auto callOp = callOps.front();
    auto yieldOp = mlir::dyn_cast<mlir::scf::YieldOp>(caseRegion.front().getTerminator());
    if (yieldOp == nullptr || yieldOp.getNumOperands() != 1 || yieldOp.getOperand(0) != callOp.getResult(0)) {
        return nullptr;
    }
    return callOp;
}

//
// AddKernelsForOverlappingEdgeTilesPass
//

class AddKernelsForOverlappingEdgeTilesPass final :
        public VPU::impl::AddKernelsForOverlappingEdgeTilesBase<AddKernelsForOverlappingEdgeTilesPass> {
public:
    explicit AddKernelsForOverlappingEdgeTilesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
    void processLoop(mlir::scf::ForOp forOp, VPU::LowResAttr lowRes);
    bool processInsert(mlir::tensor::InsertSliceOp insertOp, mlir::scf::ForOp forOp, VPU::LowResAttr lowRes,
                       mlir::Value lowResPred, Logger log);
    void removeSingleTileAsserts();
    void addTileOneCases(mlir::scf::IndexSwitchOp switchOp, ArrayRef<unsigned> fullBlkCaseIdxs,
                         mlir::tensor::InsertSliceOp insertOp, int64_t outputDim, mlir::Value tileOffset,
                         VPU::LowResAttr lowRes, mlir::Value lowResPred, Logger log);
    mlir::func::FuncOp getTileOneKernel(mlir::func::FuncOp kernel, mlir::RankedTensorType prevType, int64_t outputDim,
                                        int64_t invalidOutputs);

    // Pre-loop low resolution predicate per id: an arith.cmpi, or an arith.ori of them if the call has several sources
    llvm::DenseMap<int64_t, mlir::Value> _predicates;
    llvm::DenseSet<int64_t> _ambiguousPredicateIds;
    // Tile-1 kernel per (kernel, output dim, invalid outputs), shared by the cases that call the same kernel
    llvm::DenseMap<std::tuple<mlir::Operation*, int64_t, int64_t>, mlir::func::FuncOp> _tileOneKernels;
    // Name of the original kernel and output dims of every tile-1 kernel. A tile-1 kernel of a tile-1 kernel (both dims
    // low resolution) is named after the original kernel with all its dims: <kernel>_overlap_tile1_d2d3
    llvm::DenseMap<mlir::Operation*, std::pair<std::string, SmallVector<int64_t>>> _tileOneKernelInfo;
    // Ids with at least one loop that can run a low resolution tile, and the ids with such a loop that was not fully
    // handled (an insert or a case was skipped): only the asserts of handled, complete ids are removed
    llvm::DenseSet<int64_t> _handledIds;
    llvm::DenseSet<int64_t> _incompleteIds;
    // Dominance of the module being processed, owned by safeRunOnModule (the pass must stay copyable)
    mlir::DominanceInfo* _domInfo = nullptr;
};

/**
 * Clone of `kernel` for low resolution tile 1 along `outputDim`. It takes one more input, %prev: the first
 * `invalidOutputs` (O_l) outputs of the tile, read from the accumulator, where tile 0 wrote them. They replace the
 * invalid outputs at the start of the tile:
 *
 *   %body = VPU.Slice %r [.., O_l, ..] [.., S - O_l, ..]
 *   %t    = VPU.Concat(%prev, %body)      // along outputDim, the type of %r
 *
 * The tile type is unchanged, so the switch keeps one result and the host keeps one insert of size S. %prev is a
 * strided view of the output, like the host-side input slices.
 * The clone is named <kernel>_overlap_tile1_d<outputDim>. If `kernel` is itself a tile-1 kernel, the dim is appended
 * to the dims of its name: <kernel>_overlap_tile1_d3d2
 */
mlir::func::FuncOp AddKernelsForOverlappingEdgeTilesPass::getTileOneKernel(mlir::func::FuncOp kernel,
                                                                           mlir::RankedTensorType prevType,
                                                                           int64_t outputDim, int64_t invalidOutputs) {
    auto key = std::make_tuple(kernel.getOperation(), outputDim, invalidOutputs);
    if (auto it = _tileOneKernels.find(key); it != _tileOneKernels.end()) {
        return it->second;
    }

    auto [originalName, dims] = _tileOneKernelInfo.lookup(kernel.getOperation());
    if (dims.empty()) {
        originalName = kernel.getName().str();
    }
    dims.push_back(outputDim);

    auto* symbolTableOp = kernel->getParentOp();
    auto baseName = originalName + "_overlap_tile1_";
    for (auto dim : dims) {
        baseName += "d" + std::to_string(dim);
    }
    auto name = baseName;
    for (int64_t suffix = 1; mlir::SymbolTable::lookupSymbolIn(symbolTableOp, name) != nullptr; ++suffix) {
        name = baseName + "_" + std::to_string(suffix);
    }
    auto tileOneKernel = kernel.clone();
    tileOneKernel.setName(name);
    mlir::OpBuilder builder(kernel);
    builder.setInsertionPointAfter(kernel);
    builder.insert(tileOneKernel);

    const auto loc = kernel.getLoc();
    const auto prevArgIdx = tileOneKernel.getNumArguments();
    auto prevArgAttrs = builder.getDictionaryAttr(
            builder.getNamedAttr(HostExec::HOST_EXEC_FUNC_ARG_DYNAMIC_STRIDES_ATTR_NAME, builder.getBoolAttr(true)));
    VPUX_THROW_WHEN(mlir::failed(tileOneKernel.insertArgument(prevArgIdx, prevType, prevArgAttrs, loc)),
                    "Failed to add the low resolution input to '{0}'", name);
    auto prevArg = tileOneKernel.getArgument(prevArgIdx);

    auto returnOp = mlir::cast<mlir::func::ReturnOp>(tileOneKernel.getBody().front().getTerminator());
    auto tile = returnOp.getOperand(0);
    auto tileShape = mlir::cast<mlir::RankedTensorType>(tile.getType()).getShape();
    SmallVector<int64_t> bodyOffsets(tileShape.size(), 0);
    bodyOffsets[outputDim] = invalidOutputs;
    auto bodySizes = to_small_vector(tileShape);
    bodySizes[outputDim] -= invalidOutputs;

    builder.setInsertionPoint(returnOp);
    auto bodyOp = VPU::SliceOp::create(builder, appendLoc(loc, "lowres_body"), tile, ArrayRef<int64_t>(bodyOffsets),
                                       ArrayRef<int64_t>(bodySizes));
    auto concatOp = VPU::ConcatOp::create(builder, appendLoc(loc, "lowres_concat"),
                                          mlir::ValueRange{prevArg, bodyOp.getResult()}, outputDim);
    VPUX_THROW_WHEN(concatOp.getType() != tile.getType(), "Tile-1 kernel '{0}': concat type {1} differs from {2}", name,
                    concatOp.getType(), tile.getType());
    returnOp->setOperand(0, concatOp.getResult());

    _tileOneKernels[key] = tileOneKernel;
    _tileOneKernelInfo[tileOneKernel.getOperation()] = {std::move(originalName), std::move(dims)};
    return tileOneKernel;
}

/**
 * Adds a tile-1 case for every FULLBLK case of the dim. Case c gets the clone c + K, K (`caseShift`) a power of two
 * above every case value, which calls the tile-1 kernel with the accumulator part [bt, bt + O_l) as %prev. The
 * selector becomes
 *
 *   sel' = sel + (lowResPred && bt != 0 ? K : 0)
 *
 * Tile 1 is the low resolution tile at bt = D - S != 0. The predicate selects FULLBLK on the dim, so c + K exists
 * whenever it is selected. Tile 0, D = S and normal resolutions keep their cases. The other cases, the default case
 * and the insert are unchanged: the switch still yields one tile of size S, inserted once.
 * If both dims are low resolution, the second dim also clones the tile-1 cases of the first. Its tile-1 kernel then
 * takes the corner [0, O_l) x [0, O_l) from its own %prev, which reads the accumulator
 */
void AddKernelsForOverlappingEdgeTilesPass::addTileOneCases(mlir::scf::IndexSwitchOp switchOp,
                                                            ArrayRef<unsigned> fullBlkCaseIdxs,
                                                            mlir::tensor::InsertSliceOp insertOp, int64_t outputDim,
                                                            mlir::Value tileOffset, VPU::LowResAttr lowRes,
                                                            mlir::Value lowResPred, Logger log) {
    const auto invalidOutputs = lowRes.getInvalidOutputLeft();
    auto tileType = mlir::cast<mlir::RankedTensorType>(switchOp.getResult(0).getType());
    auto prevShape = to_small_vector(tileType.getShape());
    prevShape[outputDim] = invalidOutputs;
    auto prevType = mlir::RankedTensorType::get(prevShape, tileType.getElementType(), tileType.getEncoding());

    const auto oldCases = to_small_vector(switchOp.getCases());
    const auto maxCase = *llvm::max_element(oldCases);
    const auto caseShift = checked_cast<int64_t>(llvm::NextPowerOf2(checked_cast<uint64_t>(maxCase)));
    auto newCases = oldCases;
    for (auto caseIdx : fullBlkCaseIdxs) {
        newCases.push_back(oldCases[caseIdx] + caseShift);
    }

    mlir::OpBuilder builder(switchOp);
    const auto loc = switchOp.getLoc();
    auto zero = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
    auto caseShiftVal = mlir::arith::ConstantIndexOp::create(builder, loc, caseShift);
    auto isNonZeroOffset = mlir::arith::CmpIOp::create(builder, loc, mlir::arith::CmpIPredicate::ne, tileOffset, zero);
    auto isTileOne = mlir::arith::AndIOp::create(builder, loc, lowResPred, isNonZeroOffset);
    auto selectorShift = mlir::arith::SelectOp::create(builder, loc, isTileOne, caseShiftVal, zero);
    auto selector = mlir::arith::AddIOp::create(builder, loc, switchOp.getArg(), selectorShift);

    auto newSwitchOp = mlir::scf::IndexSwitchOp::create(builder, loc, switchOp.getResultTypes(), selector.getResult(),
                                                        newCases, newCases.size());
    newSwitchOp->setDiscardableAttrs(switchOp->getDiscardableAttrDictionary());
    newSwitchOp.getDefaultRegion().takeBody(switchOp.getDefaultRegion());
    for (auto caseIdx : irange(oldCases.size())) {
        newSwitchOp.getCaseRegions()[caseIdx].takeBody(switchOp.getCaseRegions()[caseIdx]);
    }

    for (auto [newIdx, caseIdx] : llvm::enumerate(fullBlkCaseIdxs)) {
        auto& tileOneRegion = newSwitchOp.getCaseRegions()[oldCases.size() + newIdx];
        mlir::IRMapping mapper;
        newSwitchOp.getCaseRegions()[caseIdx].cloneInto(&tileOneRegion, mapper);

        auto callOp = getCallOpFromCaseRegion(tileOneRegion);
        auto funcOp = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(callOp, callOp.getCalleeAttr());
        auto tileOneKernel = getTileOneKernel(funcOp, prevType, outputDim, invalidOutputs);

        mlir::OpBuilder caseBuilder(callOp);
        auto prevSizes = llvm::to_vector(llvm::map_range(prevShape, [&](int64_t dimSize) -> mlir::OpFoldResult {
            return caseBuilder.getIndexAttr(dimSize);
        }));
        auto prevOp = mlir::tensor::ExtractSliceOp::create(caseBuilder, appendLoc(callOp.getLoc(), "lowres_prev"),
                                                           prevType, insertOp.getDest(), insertOp.getMixedOffsets(),
                                                           prevSizes, insertOp.getMixedStrides());
        auto operands = llvm::to_vector(callOp.getOperands());
        operands.push_back(prevOp.getResult());
        auto tileOneCallOp = mlir::func::CallOp::create(caseBuilder, callOp.getLoc(), tileOneKernel, operands);
        tileOneCallOp->setDiscardableAttrs(callOp->getDiscardableAttrDictionary());
        callOp.replaceAllUsesWith(tileOneCallOp.getResults());
        callOp.erase();
        log.trace("Case {0} -> tile-1 case {1}: '{2}'", oldCases[caseIdx], newCases[oldCases.size() + newIdx],
                  tileOneKernel.getName());
    }

    switchOp.replaceAllUsesWith(newSwitchOp.getResults());
    switchOp.erase();
}

/**
 * Adjusts one output tile insert of a loop that can run a low resolution tile:
 * - the output dim of the insert is the dim whose offset is computed from the loop induction variable; its offset is
 *   the tile offset bt (after backtracking), and its size must be the step S
 * - the tagged FULLBLK input slices of the switch that produces the tile get the offset bt on the input dim of the
 *   attribute (same-scale condition, checked in AdjustBlockSizeForScfTiling). Their tiling offset max(0, bt - halo)
 *   is right only for bt = 0
 * - every FULLBLK case gets a tile-1 case (addTileOneCases)
 * Returns false if the insert is skipped: the 2-tile range is then not handled for the dim
 */
bool AddKernelsForOverlappingEdgeTilesPass::processInsert(mlir::tensor::InsertSliceOp insertOp, mlir::scf::ForOp forOp,
                                                          VPU::LowResAttr lowRes, mlir::Value lowResPred, Logger log) {
    auto switchOp = getIndexSwitchOp(insertOp, forOp);
    if (switchOp == nullptr || switchOp.getNumResults() != 1) {
        log.trace("Output insert at {0} is not fed by a single-result switch, skipped", insertOp.getLoc());
        return false;
    }

    SmallVector<int64_t> outputDims;
    for (auto [dimIdx, offset] : llvm::enumerate(insertOp.getMixedOffsets())) {
        auto offsetValue = mlir::dyn_cast<mlir::Value>(offset);
        if (offsetValue != nullptr && dependsOnInductionVar(offsetValue, forOp)) {
            outputDims.push_back(checked_cast<int64_t>(dimIdx));
        }
    }
    if (outputDims.size() != 1) {
        log.trace("Output insert: {0} output dims depend on the loop induction variable, 1 expected, skipped",
                  outputDims.size());
        return false;
    }
    const auto outputDim = outputDims.front();
    const auto tileOffset = mlir::cast<mlir::Value>(insertOp.getMixedOffsets()[outputDim]);
    const auto tileSize = mlir::getConstantIntValue(insertOp.getMixedSizes()[outputDim]);
    if (tileSize != std::optional<int64_t>(lowRes.getStep())) {
        log.trace("Output insert: size {0} on output dim {1} differs from the step {2}, skipped",
                  optionalToString(tileSize), outputDim, lowRes.getStep());
        return false;
    }

    // The tile-1 cases read the accumulator at the insert offsets, and the selector uses bt and the predicate
    auto dominatesSwitch = [&](mlir::Value value) {
        return _domInfo->properlyDominates(value, switchOp);
    };
    const auto offsetsDominate = llvm::all_of(insertOp.getMixedOffsets(), [&](mlir::OpFoldResult offset) {
        auto offsetValue = mlir::dyn_cast<mlir::Value>(offset);
        return offsetValue == nullptr || dominatesSwitch(offsetValue);
    });
    if (!offsetsDominate || !dominatesSwitch(insertOp.getDest()) || !dominatesSwitch(lowResPred)) {
        log.trace("Output insert: the insert offsets, dest or the low resolution predicate do not dominate the "
                  "switch, skipped");
        return false;
    }

    // FULLBLK cases of the dim: the case regions with a slice tagged with the id (never the default case)
    // TODO: Derive the FULLBLK cases from the caseId
    SmallVector<unsigned> fullBlkCaseIdxs;
    SmallVector<mlir::tensor::ExtractSliceOp> taggedSlices;
    for (auto [caseIdx, caseRegion] : llvm::enumerate(switchOp.getCaseRegions())) {
        const auto numTagged = taggedSlices.size();
        caseRegion.walk([&](mlir::tensor::ExtractSliceOp sliceOp) {
            if (hasLowResSliceId(sliceOp, lowRes.getId())) {
                taggedSlices.push_back(sliceOp);
            }
        });
        if (taggedSlices.size() == numTagged) {
            continue;
        }
        auto callOp = getCallOpFromCaseRegion(caseRegion);
        if (callOp == nullptr ||
            mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(callOp, callOp.getCalleeAttr()) == nullptr) {
            log.trace("Output insert: FULLBLK case {0} does not yield the tile of a single call, skipped",
                      switchOp.getCases()[caseIdx]);
            return false;
        }
        fullBlkCaseIdxs.push_back(checked_cast<unsigned>(caseIdx));
    }

    int64_t numMoved = 0;
    for (auto sliceOp : taggedSlices) {
        if (mlir::dyn_cast<mlir::Value>(sliceOp.getMixedOffsets()[lowRes.getDim()]) != tileOffset) {
            sliceOp = setSliceOffset(sliceOp, lowRes.getDim(), tileOffset);
            ++numMoved;
        }
        removeLowResSliceId(sliceOp, lowRes.getId());
    }
    log.trace("Output insert: output dim {0}, size {1}; FULLBLK input slices at bt on input dim {2}: {3} of {4} "
              "moved",
              outputDim, tileSize.value(), lowRes.getDim(), numMoved, taggedSlices.size());

    if (lowRes.getInvalidOutputLeft() == 0 || fullBlkCaseIdxs.empty()) {
        log.nest().trace("No invalid outputs at the start of tile 1 or no FULLBLK case, no tile-1 case");
        return true;
    }
    addTileOneCases(switchOp, fullBlkCaseIdxs, insertOp, outputDim, tileOffset, lowRes, lowResPred, log.nest());
    log.nest().trace("{0} tile-1 cases on output dim {1}: {2} outputs from the accumulator, {3} from the kernel",
                     fullBlkCaseIdxs.size(), outputDim, lowRes.getInvalidOutputLeft(),
                     lowRes.getStep() - lowRes.getInvalidOutputLeft());
    return true;
}

void AddKernelsForOverlappingEdgeTilesPass::processLoop(mlir::scf::ForOp forOp, VPU::LowResAttr lowRes) {
    const auto id = lowRes.getId();
    auto log = _log.nest();
    const auto step = VPU::OpChainAnalysis().getIntegerFromValue(forOp.getStep(), true);
    if (!step.has_value() || step.value() != lowRes.getStep()) {
        log.trace("Low resolution id {0} (dim {1}): loop step {2} differs from the step {3}, unrolled main loop "
                  "skipped",
                  id, lowRes.getDim(), optionalToString(step), lowRes.getStep());
        return;
    }
    auto predIt = _predicates.find(id);
    _handledIds.insert(id);
    if (predIt == _predicates.end()) {
        log.trace("Low resolution id {0} (dim {1}): no predicate, loop skipped", id, lowRes.getDim());
        _incompleteIds.insert(id);
        return;
    }
    log.trace("Low resolution id {0} (dim {1}): loop with step {2}", id, lowRes.getDim(), step.value());

    // Collected first: the switches under the loop are replaced
    SmallVector<mlir::tensor::InsertSliceOp> insertOps;
    forOp->walk([&](mlir::tensor::InsertSliceOp insertOp) {
        insertOps.push_back(insertOp);
    });
    if (insertOps.empty()) {
        _incompleteIds.insert(id);
    }
    for (auto insertOp : insertOps) {
        if (!processInsert(insertOp, forOp, lowRes, predIt->second, log.nest())) {
            _incompleteIds.insert(id);
        }
    }
}

/**
 * Removes the single-tile assert of AdjustBlockSizeForScfTiling (assert(!lowRes || upperBound == step)) for the dims
 * whose 2-tile range is now handled in every loop that can run a low resolution tile. The asserts of the other dims
 * stay, so their 2-tile range keeps failing at runtime instead of computing wrong outputs
 */
void AddKernelsForOverlappingEdgeTilesPass::removeSingleTileAsserts() {
    SmallVector<mlir::cf::AssertOp> assertOps;
    getOperation()->walk([&](mlir::cf::AssertOp assertOp) {
        if (assertOp->hasAttr(VPU::LOW_RES_ASSERT_ATTR_NAME)) {
            assertOps.push_back(assertOp);
        }
    });
    for (auto assertOp : assertOps) {
        const auto id = assertOp->getAttrOfType<mlir::IntegerAttr>(VPU::LOW_RES_ASSERT_ATTR_NAME).getInt();
        if (!_handledIds.contains(id) || _incompleteIds.contains(id)) {
            _log.trace("Low resolution id {0}: 2-tile range not handled in every loop, single-tile assert kept", id);
            assertOp->removeAttr(VPU::LOW_RES_ASSERT_ATTR_NAME);
            continue;
        }
        // The condition select(lowRes, upperBound == step, true) is used only by the assert; the predicate stays
        SmallVector<mlir::Operation*> worklist{assertOp.getArg().getDefiningOp()};
        llvm::SmallPtrSet<mlir::Operation*, 8> erased;
        assertOp.erase();
        while (!worklist.empty()) {
            auto* op = worklist.pop_back_val();
            if (op == nullptr || erased.contains(op) || !op->use_empty() || !mlir::isPure(op) ||
                op->hasAttr(VPU::LOW_RES_PRED_ATTR_NAME)) {
                continue;
            }
            for (auto operand : op->getOperands()) {
                worklist.push_back(operand.getDefiningOp());
            }
            erased.insert(op);
            op->erase();
        }
        _log.trace("Low resolution id {0}: single-tile assert removed", id);
    }
}

void AddKernelsForOverlappingEdgeTilesPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    _predicates.clear();
    _ambiguousPredicateIds.clear();
    _tileOneKernels.clear();
    _tileOneKernelInfo.clear();
    _handledIds.clear();
    _incompleteIds.clear();
    moduleOp->walk([&](mlir::Operation* op) {
        auto idAttr = op->getAttrOfType<mlir::IntegerAttr>(VPU::LOW_RES_PRED_ATTR_NAME);
        if (idAttr == nullptr) {
            return;
        }
        // Several predicates of one id (several calls in the loop) are not the same condition: the id is not handled
        if (_ambiguousPredicateIds.contains(idAttr.getInt())) {
            return;
        }
        if (!_predicates.try_emplace(idAttr.getInt(), op->getResult(0)).second) {
            _predicates.erase(idAttr.getInt());
            _ambiguousPredicateIds.insert(idAttr.getInt());
        }
    });

    SmallVector<std::pair<mlir::scf::ForOp, VPU::LowResAttr>> loops;
    moduleOp->walk([&](mlir::scf::ForOp forOp) {
        if (auto lowRes = forOp->getAttrOfType<VPU::LowResAttr>(VPU::LOW_RES_ATTR_NAME)) {
            loops.emplace_back(forOp, lowRes);
        }
    });
    _log.trace("{0} low resolution predicates, {1} low resolution loops", _predicates.size(), loops.size());
    for (auto [forOp, lowRes] : loops) {
        // Recomputed per loop: the switches of the previous loops were replaced
        mlir::DominanceInfo domInfo(moduleOp);
        _domInfo = &domInfo;
        processLoop(forOp, lowRes);
        _domInfo = nullptr;
    }

    removeSingleTileAsserts();

    // Tags left are FULLBLK slices of loops that cannot run a low resolution tile (unrolled main loops), or of inserts
    // that were skipped above
    int64_t numLeft = 0;
    moduleOp->walk([&](mlir::tensor::ExtractSliceOp sliceOp) {
        if (sliceOp->hasAttr(VPU::LOW_RES_SLICE_ATTR_NAME)) {
            sliceOp->removeAttr(VPU::LOW_RES_SLICE_ATTR_NAME);
            ++numLeft;
        }
    });
    _log.trace("{0} tile-1 kernels, {1} tagged FULLBLK slices left unchanged", _tileOneKernels.size(), numLeft);

    // The predicates and the loop data are consumed by this pass
    moduleOp->walk([&](mlir::Operation* op) {
        op->removeAttr(VPU::LOW_RES_PRED_ATTR_NAME);
        op->removeAttr(VPU::LOW_RES_ATTR_NAME);
    });
}

}  // namespace

//
// createAddKernelsForOverlappingEdgeTilesPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createAddKernelsForOverlappingEdgeTilesPass(Logger log) {
    return std::make_unique<AddKernelsForOverlappingEdgeTilesPass>(log);
}
