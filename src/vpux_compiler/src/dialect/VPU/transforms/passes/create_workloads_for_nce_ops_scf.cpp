//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_analysis_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_iteration_shape_resolver.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/workload_split_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <llvm/Support/MathExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

#include <vpu_cost_model.h>

#include <algorithm>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace vpux::VPU {
#define GEN_PASS_DECL_WORKLOADSFORNCEOPSSCF
#define GEN_PASS_DEF_WORKLOADSFORNCEOPSSCF
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

/**
 * When per-iteration shape resolution fails, the pass falls back to the op's declared bound.
 *
 * That is harmless when the bound already describes a single tile - the documented behaviour
 * for a genuinely dynamic op such as @ConvInsideForallDynamicH, whose bound is the per-tile
 * upper bound rather than the size of the whole tensor.
 *
 * It is harmful when the bound describes the *whole* loop-carried tensor: the workload then
 * claims the entire tensor on every iteration, oversized by roughly the tiling factor. Detect
 * exactly that by comparing the fallback bound against the tensors carried by the enclosing
 * loops, and return the factor by which the workload is inflated so it can be reported.
 *
 * Returns nullopt when the fallback is benign or cannot be shown to be oversized.
 */
std::optional<int64_t> getFallbackOversizeFactor(VPU::NCEOpInterface nceOp, ArrayRef<int64_t> bounds) {
    const auto carriesWholeTensor = [&](mlir::ValueRange carried) {
        return llvm::any_of(carried, [&](mlir::Value operand) {
            if (mlir::dyn_cast<vpux::NDTypeInterface>(operand.getType()) == nullptr) {
                return false;
            }
            return getBoundedShape(operand.getType()).raw() == bounds;
        });
    };

    for (auto* current = nceOp->getParentOp(); current != nullptr; current = current->getParentOp()) {
        mlir::ValueRange carried;
        if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(current)) {
            carried = forOp.getInitArgs();
        } else if (auto forallOp = mlir::dyn_cast<mlir::scf::ForallOp>(current)) {
            carried = forallOp.getOutputs();
        } else {
            continue;
        }

        // Only a loop that actually splits the tensor can make the fallback oversized.
        // Unknown trip counts are ignored and only affect the diagnostic below.
        const auto tripCount = VPU::getConstantTripCount(current);
        if (!tripCount.has_value() || tripCount.value() <= 1) {
            continue;
        }
        if (carriesWholeTensor(carried)) {
            return tripCount;
        }
    }
    return std::nullopt;
}

/**
 * Per-iteration tile shapes of one NCE op: its output shape plus the shape of EVERY operand.
 *
 * Border-tile distinction: two iterations may produce the same
 * output shape while their input slice differs (e.g. tile at the H/W boundary vs. an
 * interior tile when boundary padding is materialized via tensor.pad on the input).
 * Such tiles must be treated as distinct: their workload cost — and thus the split
 * decision — depends on the input size, not only the output size. Mirrors what
 * `getUniqueShapeTilingCandidates` does in the default (non-SCF) tiling path.
 *
 * All operands are carried, not just the activation: workload correction and input-workload
 * back-inference read the shapes of every operand, so a multi-operand op (e.g. the
 * VPU.NCE.Eltwise closing a Vertical Fusion skip connection) needs each of them pinned to the
 * same iteration. Mixing a per-tile activation with a whole-tensor second operand yields input
 * workloads sized for the whole tensor.
 */
struct TileShapes {
    Shape outputShape;
    /// Indexed by operand number. Entries for operands without an NDTypeInterface type are empty.
    SmallVector<Shape> operandShapes;

    /// Activation (operand 0) shape — what the cost params and the runtime dispatch key on.
    const Shape& inputShape() const {
        VPUX_THROW_WHEN(operandShapes.empty(), "TileShapes carries no operand shapes");
        return operandShapes.front();
    }
};

/**
 * A dynamic dim of one NCE operand — a quantity the runtime dispatch conditions can compare
 * against. Operands cover every tile difference there is: the NCE output shape is a function of
 * the operand shapes, so two tiles that differ at all differ on some operand dim.
 */
struct OperandExtent {
    unsigned operandIdx;
    int64_t dim;
    // The size this dim held in each enumerated tile, indexed like TileShapeEnumeration::tiles.
    // These are the values the dispatch conditions compare the runtime extent against.
    SmallVector<int64_t> tileSizes;
};

/**
 * Outcome of the per-iteration tile enumeration: the tiles themselves plus the operand extents
 * a caller can key a runtime dispatch on, so it does not have to walk the tiles again.
 */
struct TileShapeEnumeration {
    // Deduplicated tile shapes; empty when enumeration failed.
    SmallVector<TileShapes> tiles;
    // Candidate dispatch keys, each carrying its own per-tile sizes.
    SmallVector<OperandExtent> dispatchExtents;
};

/**
 * Choose the smallest prefix-greedy set of extents whose sizes identify every enumerated tile
 * uniquely, returned as indices into TileShapeEnumeration::dispatchExtents.
 *
 * Returns {} when no set of extents identifies every tile: two tiles would match the same
 * condition, so one of them would silently run the other's workloads.
 */
SmallVector<size_t> selectDispatchKeys(ArrayRef<OperandExtent> extents, size_t numTiles) {
    // Tiles whose sizes agree on every picked extent are indistinguishable and would match the
    // same condition. Counting the resulting groups says how many tiles a set of extents resolves,
    // which reaches numTiles exactly when the set identifies each tile on its own.
    const auto countDistinguished = [&](ArrayRef<size_t> picked) {
        std::set<std::vector<int64_t>> groups;
        for (size_t tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
            std::vector<int64_t> sizes;
            sizes.reserve(picked.size());
            for (auto extentIdx : picked) {
                sizes.push_back(extents[extentIdx].tileSizes[tileIdx]);
            }
            groups.insert(std::move(sizes));
        }
        return groups.size();
    };

    SmallVector<size_t> picked;
    auto distinguished = countDistinguished(picked);
    for (size_t extentIdx = 0; extentIdx < extents.size() && distinguished < numTiles; ++extentIdx) {
        picked.push_back(extentIdx);
        const auto withExtent = countDistinguished(picked);
        if (withExtent == distinguished) {
            // Splits no group the already picked extents did not, so it would only add a redundant
            // condition - and can be outright wrong to compare against: a dynamic dim the loop
            // never tiles is enumerated as the type's bound for every tile, while at runtime it
            // holds the actual size, which would make every condition false at once.
            picked.pop_back();
            continue;
        }
        distinguished = withExtent;
    }

    return distinguished == numTiles ? std::move(picked) : SmallVector<size_t>{};
}

/**
 * Declared per-operand shapes: bounds for the dynamic dimensions of a bounded-dynamic operand,
 * the static shape otherwise. Used when per-iteration resolution is not needed (static output)
 * or not possible (fallback path), so that all operands are described the same way in every case.
 */
SmallVector<Shape> getDeclaredOperandShapes(VPU::NCEOpInterface nceOp) {
    SmallVector<Shape> shapes;
    shapes.reserve(nceOp->getNumOperands());
    for (auto operand : nceOp->getOperands()) {
        if (mlir::dyn_cast<vpux::NDTypeInterface>(operand.getType()) == nullptr) {
            shapes.push_back(Shape{});
            continue;
        }
        shapes.push_back(getShapeUsingBoundsForDynamicDims(operand.getType()));
    }
    return shapes;
}

/**
 * Enumerate the unique per-iteration (output, input) tile-shape pairs for an NCE op
 * that lives inside one or more scf.for / scf.forall loops.
 *
 * Walks the Cartesian product of all parent loop iterations and, for each iteration,
 * resolves the concrete output and input shapes with `IterationShapeResolver`. The
 * resolver derives the output shape either from the insertion op the result is written
 * into (back-inferring through any shape transform on the way there, e.g. VPU.DepthToSpace)
 * or — for the intermediate ops of a Vertical Fusion chain, which feed the next NCE op
 * instead of an insert_slice — by forward inference from the resolved input shape.
 * Results are deduplicated by the (output, input) pair so that border tiles whose output
 * matches a middle tile but whose input differs are preserved as distinct entries.
 *
 * Every tile also carries the sizes of the dynamic operand dims it was enumerated with, so a
 * caller can key a runtime dispatch on them without walking the same ops again.
 *
 * Returns an empty tile list on any failure (cannot find loops / ops, OFR evaluation
 * fails, or iteration count would exceed the cap).
 *
 * TODO: E#222387 — Refactor to reuse the IV collection and iteration-space evaluation
 * infrastructure from scf_loop_analysis_and_debug / scf_analyzer (collectIVArgs,
 * constructIterationSpace, evaluateOpValuePerIteration). The scf_analyzer path needs
 * scf.forall support first.
 */
TileShapeEnumeration getUniqueDynamicTileShapes(VPU::NCEOpInterface nceOp, Logger log) {
    auto outputType = nceOp->getResult(0).getType();
    auto outBounded = mlir::dyn_cast<vpux::Core::BoundedTensorType>(outputType);
    if (!outBounded) {
        return {};
    }
    const auto outShapeView = mlir::cast<vpux::NDTypeInterface>(outputType).getShape();

    // Identify which dimensions are dynamic for the output.
    SmallVector<int64_t> outDynDims;
    for (int64_t i = 0; i < static_cast<int64_t>(outShapeView.size()); ++i) {
        if (outShapeView[Dim(i)] == mlir::ShapedType::kDynamic) {
            outDynDims.push_back(i);
        }
    }

    if (outDynDims.empty()) {
        // TODO: #E220233 — Support position-dependent DPU workload padding for halo-extracted tiles
        // When the output shape is fully static, all tiles share the same output shape but may
        // still need different effective DPU padding depending on their position in the spatial
        // grid (first/middle/last along H × W -> up to 9 unique workload configurations).
        // This applies only when the SCF tiling infrastructure uses halo-based input extraction
        // (input tile is sized to provide halo data from adjacent tiles, so interior tiles can
        // run with pad=0). The current SCF flow does NOT use halo-based extraction:
        //   - With tensor.pad: NCE pad is already 0 -> all positions equivalent.
        //   - Without tensor.pad: each tile applies its own NCE pad -> all positions equivalent.
        // Implementing position-dependent workloads here is a no-op for the current flow and
        // would require a halo-based input-tile detection (e.g., inputTile_H == outputTile_H +
        // (kernel_H - 1)) plus runtime offset comparisons in the workloads region.
        return {};
    }

    // Cap the total number of (output, input) tile pairs we may enumerate. Enforced against
    // the real Cartesian-product iteration count of the parent loops below.
    constexpr int64_t kMaxGeneratedShapes = 4096;

    // Collect parent scf.for / scf.forall loops; build ordered IV list with per-IV value sequences.
    OpChainAnalysis opAnalysis(log);
    SmallVector<mlir::Value> ivOrder;
    SmallVector<SmallVector<int64_t>> ivValuesOrder;

    auto* current = nceOp->getParentOp();
    while (current) {
        if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(current)) {
            auto [lb, ub, step] = opAnalysis.getLoopBoundsAndStep(forOp);

            SmallVector<int64_t> ivValues;
            for (int64_t iv = lb; iv < ub; iv += step) {
                ivValues.push_back(iv);
            }
            if (ivValues.empty()) {
                // Zero-iteration loop: no per-iteration shapes to enumerate.
                return {};
            }
            ivOrder.push_back(forOp.getInductionVar());
            ivValuesOrder.push_back(std::move(ivValues));
        } else if (auto forallOp = mlir::dyn_cast<mlir::scf::ForallOp>(current)) {
            auto mixedLB = forallOp.getMixedLowerBound();
            auto mixedUB = forallOp.getMixedUpperBound();
            auto mixedStep = forallOp.getMixedStep();
            auto inductionVars = forallOp.getInductionVars();
            for (size_t dim = 0; dim < inductionVars.size(); ++dim) {
                auto lb = mlir::getConstantIntValue(mixedLB[dim]);
                auto ub = mlir::getConstantIntValue(mixedUB[dim]);
                auto step = mlir::getConstantIntValue(mixedStep[dim]);
                if (!lb || !ub || !step || *step <= 0) {
                    log.trace("Cannot enumerate scf.forall induction var #{0} for op at '{1}': "
                              "non-constant bounds/step",
                              dim, nceOp->getLoc());
                    return {};
                }
                SmallVector<int64_t> ivValues;
                for (int64_t iv = *lb; iv < *ub; iv += *step) {
                    ivValues.push_back(iv);
                }
                if (ivValues.empty()) {
                    return {};
                }
                ivOrder.push_back(inductionVars[dim]);
                ivValuesOrder.push_back(std::move(ivValues));
            }
        }
        current = current->getParentOp();
    }

    if (ivOrder.empty()) {
        log.trace("NCE op not inside any scf.for/scf.forall — cannot derive unique tile shapes");
        return {};
    }

    // Total iteration count = product of all IV value counts; bail if it would exceed the cap.
    int64_t totalIterations = 1;
    for (const auto& vals : ivValuesOrder) {
        int64_t sz = vals.empty() ? 1 : static_cast<int64_t>(vals.size());
        if (totalIterations > kMaxGeneratedShapes / sz) {
            log.trace("Skip tile-shape enumeration for '{0}': iteration count ({1}) exceeds cap ({2})",
                      nceOp->getName(), totalIterations * sz, kMaxGeneratedShapes);
            return {};
        }
        totalIterations *= sz;
    }

    // Dedupe on the flat concatenation of the output shape and every operand shape. All ranks are
    // fixed for a given op, so the concatenation is an unambiguous key; std::map keeps the
    // enumeration order deterministic regardless of loop-nest traversal order.
    std::map<std::vector<int64_t>, TileShapes> tileMap;

    SmallVector<size_t> idx(ivOrder.size(), 0);
    while (true) {
        // Build a single-valued IV map for this iteration.
        ValueRangeMap singleMap;
        for (size_t i = 0; i < ivOrder.size(); ++i) {
            singleMap[ivOrder[i]] = {ivValuesOrder[i][idx[i]]};
        }

        // Resolve the output shape for the current iteration. The resolver handles both the
        // standalone case (result written into an insert_slice) and the Vertical Fusion case
        // (result consumed by the next NCE op of the chain).
        IterationShapeResolver resolver(opAnalysis, singleMap, log);
        auto outShape = resolver.getNceOutputShape(nceOp);
        if (!outShape.has_value()) {
            // Same reasoning as the unresolvable-operand case below: a tile set missing one tile is
            // worse than no tile set at all.
            log.trace("Output shape of the op at '{0}' cannot be pinned to a single iteration; "
                      "abandoning per-iteration enumeration",
                      nceOp->getLoc());
            return {};
        }

        // Resolve every operand against the same iteration, not only the activation. Statically
        // shaped operands (weights, weight tables) resolve to their own shape immediately.
        SmallVector<Shape> operandShapes;
        for (const auto& [operandIdx, operand] : nceOp->getOpOperands() | indexed) {
            if (mlir::dyn_cast<vpux::NDTypeInterface>(operand.get().getType()) == nullptr) {
                operandShapes.push_back(Shape{});
                continue;
            }
            auto shape = resolver.getValueShape(operand.get());
            if (!shape.has_value()) {
                // A dynamic operand that cannot be pinned to this iteration would have to be
                // described by its full bound, which inside a loop is typically the whole
                // loop-carried tensor - an input workload sized for the entire tensor behind a
                // correct-looking per-tile output shape.
                //
                // Abandon the whole enumeration rather than just this iteration: the runtime
                // dispatch built from the result is exhaustive over the tiles it knows about, its
                // last entry being an unconditional else. A set missing one tile therefore does not
                // degrade to a bound, it silently routes that tile to another tile's workloads.
                // Returning empty takes the bounded fallback, which is reported via
                // getFallbackOversizeFactor.
                log.trace("Operand {0} ({1}) of the op at '{2}' cannot be pinned to a single iteration; "
                          "abandoning per-iteration enumeration",
                          operandIdx, operand.get().getType(), nceOp->getLoc());
                return {};
            }
            operandShapes.push_back(std::move(shape.value()));
        }

        std::vector<int64_t> key(outShape->raw().begin(), outShape->raw().end());
        for (const auto& operandShape : operandShapes) {
            key.insert(key.end(), operandShape.raw().begin(), operandShape.raw().end());
        }
        tileMap.try_emplace(std::move(key), TileShapes{outShape.value(), std::move(operandShapes)});

        // Increment Cartesian-product index.
        size_t pos = 0;
        while (pos < idx.size()) {
            if (++idx[pos] < ivValuesOrder[pos].size()) {
                break;
            }
            idx[pos] = 0;
            ++pos;
        }
        if (pos == idx.size()) {
            break;
        }
    }

    if (tileMap.empty()) {
        log.trace("Failed to derive per-iteration tile shapes for dynamic NCE op at '{0}'", nceOp->getLoc());
        return {};
    }

    SmallVector<TileShapes> result;
    result.reserve(tileMap.size());
    for (auto& entry : tileMap) {
        result.push_back(std::move(entry.second));
    }

    // Sort descending by output total elements (largest first), then by input total elements.
    // stable_sort so that entries tied on both keep the map's deterministic key order.
    llvm::stable_sort(result, [](const TileShapes& a, const TileShapes& b) {
        int64_t ta = 1, tb = 1;
        for (auto v : a.outputShape.raw()) {
            ta *= v;
        }
        for (auto v : b.outputShape.raw()) {
            tb *= v;
        }
        if (ta != tb) {
            return ta > tb;
        }
        int64_t ia = 1, ib = 1;
        for (auto v : a.inputShape().raw()) {
            ia *= v;
        }
        for (auto v : b.inputShape().raw()) {
            ib *= v;
        }
        return ia > ib;
    });

    TileShapeEnumeration enumeration;
    enumeration.tiles = std::move(result);

    // Every dynamic dim of every operand is a candidate dispatch key: the tile a runtime iteration
    // belongs to is fully determined by its operand shapes, so some subset of these extents is
    // guaranteed to tell every enumerated tile apart, if any subset can.
    for (const auto& [operandIdx, operand] : llvm::enumerate(nceOp->getOperands())) {
        auto operandNdType = mlir::dyn_cast<vpux::NDTypeInterface>(operand.getType());
        if (operandNdType == nullptr) {
            continue;
        }
        const auto operandShapeView = operandNdType.getShape();
        for (int64_t dim = 0; dim < static_cast<int64_t>(operandShapeView.size()); ++dim) {
            if (operandShapeView[Dim(dim)] != mlir::ShapedType::kDynamic) {
                continue;
            }
            OperandExtent extent{static_cast<unsigned>(operandIdx), dim, {}};
            extent.tileSizes.reserve(enumeration.tiles.size());
            for (const auto& tile : enumeration.tiles) {
                extent.tileSizes.push_back(tile.operandShapes[operandIdx][Dim(dim)]);
            }
            enumeration.dispatchExtents.push_back(std::move(extent));
        }
    }

    // A tile set the runtime cannot dispatch between is worse than no tile set at all: the
    // materialization below is exhaustive over whatever tiles it is given, so an ambiguous pair
    // would silently route one tile's iterations to the other's workloads rather than degrading to
    // the (correct, if oversized) bounded fallback.
    if (enumeration.tiles.size() > 1 &&
        selectDispatchKeys(enumeration.dispatchExtents, enumeration.tiles.size()).empty()) {
        log.trace("Op at '{0}' has {1} distinct tiles that no combination of operand extents tells apart; the "
                  "runtime dispatch cannot separate them. Abandoning per-iteration enumeration",
                  nceOp->getLoc(), enumeration.tiles.size());
        return {};
    }

    log.trace("Found {0} unique (output, operands) tile shapes for op at '{1}'", enumeration.tiles.size(),
              nceOp->getLoc());
    return enumeration;
}

/**
 * Per-tile workload computation results: descriptors plus the (output, input) shape
 * this entry was computed for. Input shape is preserved so runtime dispatch can key
 * on it (border tiles may share output shape with interior tiles but differ on input).
 */
struct ShapeWorkloads {
    Shape outputShape;
    Shape inputShape;
    SmallVector<VPU::WorkloadDescriptor, 4> descriptors;
    SmallVector<int64_t> supportedChannels;  // from correctWorkloadDescriptors
    int64_t dpuCost = 0;                     // cost from VPUNN for this shape
};

/**
 * Compute workloads for the given static output shape in the SCF flow.
 * This helper does not mutate the NCE op's output or input types. Instead, it uses
 * the provided shape together with SCF-specific workload-cost helpers to derive the
 * workload information for this shape.
 *
 * Pad handling and MC strategy are both derived inside getWorkloadCostParamForSCF
 * (see cost_model.cpp): pad uses the op's pad attribute (consistency-checked against any
 * tensor.pad on the activation); MC strategy is inferred from the enclosing scf.forall
 * loop structure; if no scf.forall found, assume op is single cluster.
 *
 * TODO: #E218003 — SEP info is not handled (needs SparseTensor support in SCF flow).
 */
ShapeWorkloads computeWorkloadsForShape(VPU::NCEOpInterface nceOp, const Shape& staticShape,
                                        ArrayRef<Shape> operandShapes, mlir::MLIRContext* ctx, config::ArchKind arch,
                                        int64_t numDPUs, VPUNN::VPUCostModel& costModel, Logger log) {
    VPUX_THROW_WHEN(operandShapes.empty(), "No per-tile operand shapes provided for op at '{0}'", nceOp->getLoc());

    ShapeWorkloads result;
    result.outputShape = staticShape;

    // --- Compute input shape ---
    // Use the explicit per-tile input shape provided by the caller (this captures
    // border-vs-interior tile differences when tensor.pad augments the extract_slice).
    // For depthwise ops, override the channel dimension to match the output.
    const bool isDepthwise = VPU::isDepthwiseOp(nceOp.getOperation());
    const auto channelSize = staticShape[Dims4D::Act::C];

    Shape inputShape = operandShapes.front();
    if (isDepthwise && inputShape.size() == 4 && inputShape[Dims4D::Act::C] != channelSize) {
        inputShape[Dims4D::Act::C] = channelSize;
    }
    // result.inputShape must match the shape actually fed to cost computation below
    // (costParams / backInferInputWorkloads), so runtime dispatch comparisons key on the
    // same value — not the pre-correction shape.
    result.inputShape = inputShape;

    // Build WorkloadCostParams directly. Pad info and MC strategy (with their consistency checks /
    // scf.forall inference) are derived inside getWorkloadCostParamForSCF
    auto costParams =
            VPU::getWorkloadCostParamForSCF(nceOp, arch, numDPUs, ShapeRef(inputShape), ShapeRef(staticShape), log);

    // Compute workload descriptors
    // Uses the pre-built cost params and explicit output shape for MPE mode computation.
    // generateWorkloadDescriptors sets DPUCost on the op as a side-effect.
    auto originalDPUCostAttr = nceOp->getAttrOfType<mlir::IntegerAttr>(DPUCost);

    int64_t dpuCost = 0;
    result.descriptors =
            VPU::computeWorkloadDescriptorsForSCF(nceOp, costParams, ShapeRef(staticShape), costModel, dpuCost, log);
    result.dpuCost = dpuCost;

    // Apply corrections and back-infer input workloads. Both take the resolved per-tile input shape
    // explicitly instead of reading it from operand(0)'s IR type, which may still be dynamically
    // shaped here (e.g. SCF flow using dynamic/bounded tensor types)
    VPU::correctWorkloadDescriptors(nceOp, result.descriptors, ctx, log, &result.supportedChannels,
                                    ShapeRef(inputShape));

    VPU::backInferInputWorkloads(nceOp, result.descriptors, log, ShapeRef(inputShape));

    // Restore original DPUCost attribute (generateWorkloadDescriptors overwrites it).
    if (originalDPUCostAttr) {
        nceOp->setAttr(DPUCost, originalDPUCostAttr);
    } else {
        nceOp->removeAttr(DPUCost);
    }

    return result;
}

//
// WorkloadsForNCEOpsSCFPass
//

class WorkloadsForNCEOpsSCFPass final : public VPU::impl::WorkloadsForNCEOpsSCFBase<WorkloadsForNCEOpsSCFPass> {
public:
    explicit WorkloadsForNCEOpsSCFPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void WorkloadsForNCEOpsSCFPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    auto module = func->getParentOfType<mlir::ModuleOp>();

    if (config::isPureHostCompileFunc(func)) {
        _log.debug("Nothing to do with a pure HostCompile function: {0}", func);
        return;
    }

    const auto arch = config::getArch(module);

    auto nceCluster = config::getTileExecutor(module);
    VPUX_THROW_WHEN(nceCluster == nullptr, "Failed to get NCE_Cluster information");

    auto dpuExec = nceCluster.getSubExecutor(config::ExecutorKind::DPU);
    VPUX_THROW_WHEN(dpuExec == nullptr, "Failed to get DPU information");
    const auto numDPUs = dpuExec.getCount();

    auto maybeCostModelAnalysis = getCachedParentAnalysis<VPU::CostModelAnalysis>(module);
    auto costModel = VPU::CostModelAnalysis::getOrCreateCostModel(maybeCostModelAnalysis, &ctx, _log);

    // Walk all NCE ops inside scf.forall / scf.for bodies
    func->walk([&](VPU::NCEOpInterface nceOp) {
        // Only handle ops that are inside an scf.forall / scf.for
        if (!nceOp->getParentOfType<mlir::scf::ForallOp>() && !nceOp->getParentOfType<mlir::scf::ForOp>()) {
            return;
        }

        // Skip if workloads already generated
        if (!nceOp.getWorkloads().empty()) {
            return;
        }

        _log.trace("SCF workload split for op '{0}' at '{1}'", nceOp->getName(), nceOp->getLoc());

        auto outputType = nceOp->getResult(0).getType();
        auto ndType = mlir::cast<vpux::NDTypeInterface>(outputType);
        const auto outputShape = ndType.getShape();

        // Determine the set of unique (output, input) tile shape pairs to compute workloads for.
        // Static output -> single pair; dynamic output -> enumerate per-iteration pairs (preserves
        // border tiles whose input slice differs from interior tiles with the same output shape).
        TileShapeEnumeration enumeration;
        auto& tileShapes = enumeration.tiles;
        if (!outputShape.isDynamic()) {
            tileShapes.push_back({outputShape.toValues(), getDeclaredOperandShapes(nceOp)});
        } else {
            auto boundedType = mlir::dyn_cast<vpux::Core::BoundedTensorType>(outputType);
            VPUX_THROW_WHEN(boundedType == nullptr, "NCE op at '{0}' has dynamic output without bounds",
                            nceOp->getLoc());

            // Enumerate per-iteration (output, input) tile shapes from parent scf.for / scf.forall
            // loops, accounting for tensor.pad on the input. Two iterations sharing an output
            // shape but differing on input are kept distinct.
            enumeration = getUniqueDynamicTileShapes(nceOp, _log);

            if (tileShapes.empty()) {
                // Fallback: use bounded output shape and bounded/static input shape.
                // This handles cases where the NCE op's output doesn't flow into an
                // insert_slice (e.g., multi-step reductions where the first NCE feeds
                // a second reduction op rather than being inserted back into an accumulator).
                //
                // When the bound is the whole loop-carried tensor rather than a single tile the
                // fallback produces workloads oversized by the tiling factor.
                //
                // Downstream, clampOversizedWorkloadsToStaticOutput() (run at the end of
                // FullUnrollSCFLoop, after the unrolled shapes have been inferred) repairs the
                // geometry of these descriptors, but only for ops left with a single workload at
                // a zero offset. Two gaps survive it: an op whose tile was split across several
                // DPU workloads is not clamped at all, and the DPUCost attribute set below is
                // never recomputed, so the cost stays the one estimated from the oversized shape.
                const auto oversizeFactor = getFallbackOversizeFactor(nceOp, boundedType.getBounds().raw());
                if (oversizeFactor.has_value()) {
                    _log.warning("Could not derive per-iteration tile shapes for op '{0}' at '{1}'. Falling back "
                                 "to the bounded shape {2}, which is the whole tensor carried by an enclosing "
                                 "{3}-iteration loop - the generated workloads are oversized by up to {3}x. "
                                 "Single-workload ops are clamped back by clampOversizedWorkloadsToStaticOutput "
                                 "after full unroll; multi-workload ops are not, and the DPU cost stays estimated "
                                 "from the oversized shape in either case.",
                                 nceOp->getName(), nceOp->getLoc(), boundedType.getBounds(), oversizeFactor.value());
                } else {
                    _log.trace("Falling back to bounded shapes for op at '{0}'", nceOp->getLoc());
                }
                tileShapes.push_back({getShapeUsingBoundsForDynamicDims(outputType), getDeclaredOperandShapes(nceOp)});
            }
        }

        _log.trace("Unique (output, input) tile shapes for op at '{0}': {1}", nceOp->getLoc(), tileShapes.size());

        // Validate DW channel decomposability. In the SCF flow, channel alignment may not
        // have been applied to ops created inside the loop body (e.g., NCEMaxPool from
        // multi-step Reduce lowering). If the per-tile channel count is not decomposable
        // into supported DW workload sizes, skip workload generation — the op's channel
        // configuration is invalid for the target hardware and must be fixed upstream.
        if (VPU::isDepthwiseOp(nceOp.getOperation())) {
            const auto& strategyFactory = VPU::getVPUStrategyFactory(&ctx);
            const auto supportedDW = strategyFactory->getSupportedChannelsDW();
            const int64_t minDW = !supportedDW.empty() ? supportedDW.back() : VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
            bool hasInvalidChannels = false;
            for (const auto& ts : tileShapes) {
                auto ch = ts.outputShape[Dims4D::Act::C];
                if (ch % minDW == 0) {
                    continue;
                }
                _log.warning("Skipping workload generation for DW op at '{0}': channel count {1} is not a "
                             "multiple of minimum supported DW channel size {2}",
                             nceOp->getLoc(), ch, minDW);
                hasInvalidChannels = true;
                break;
            }
            if (hasInvalidChannels) {
                return;
            }
        }

        // Compute workloads for each unique tile shape pair.
        SmallVector<ShapeWorkloads, 2> perShapeWorkloads;
        for (const auto& [shapeIdx, ts] : llvm::enumerate(tileShapes)) {
            _log.trace("Computing workloads for tile shape [{0}/{1}]: outputShape={2}, operandShapes={3}", shapeIdx + 1,
                       tileShapes.size(), ts.outputShape, ts.operandShapes);
            auto sw = computeWorkloadsForShape(nceOp, ts.outputShape, ts.operandShapes, &ctx, arch, numDPUs, *costModel,
                                               _log);
            _log.trace("  -> descriptors={0}, dpuCost={1}, supportedChannels={2}", sw.descriptors.size(), sw.dpuCost,
                       sw.supportedChannels);
            perShapeWorkloads.push_back(std::move(sw));
        }

        // TODO: #E215880 — implement shiftWorkloadsForHalo once MC-in-SCF is fully fleshed out

        // Materialize workloads.
        mlir::OpBuilder builder(&ctx);
        builder.setInsertionPointAfter(nceOp);

        // Helper to emit DPU.Workload ops from a descriptor list.
        auto emitWorkloads = [&](mlir::OpBuilder& b, ArrayRef<VPU::WorkloadDescriptor> descriptors) {
            for (const auto& wl : descriptors) {
                auto mpeModeAttr = VPU::MPEModeAttr::get(&ctx, wl.mpeMode);
                mlir::IntegerAttr clusterIdAttr =
                        wl.clusterId.has_value() ? getIntAttr(&ctx, wl.clusterId.value()) : nullptr;
                auto padAttr = VPU::getPaddingAttr(&ctx, wl.padding);

                auto outOffsetsAttr = mlir::DenseI64ArrayAttr::get(&ctx, wl.outOffsets.raw());
                auto outSizesAttr = mlir::DenseI64ArrayAttr::get(&ctx, wl.outSizes.raw());

                if (wl.inOffsets.empty() || wl.inSizes.empty()) {
                    VPU::DPUWorkloadOp::create(b, nceOp->getLoc(), outOffsetsAttr, outSizesAttr, padAttr, mpeModeAttr,
                                               clusterIdAttr);
                    continue;
                }
                auto inOffsetsAttr = mlir::DenseI64ArrayAttr::get(&ctx, wl.inOffsets.raw());
                auto inSizesAttr = mlir::DenseI64ArrayAttr::get(&ctx, wl.inSizes.raw());
                VPU::DPUWorkloadOp::create(b, nceOp->getLoc(), outOffsetsAttr, outSizesAttr, inOffsetsAttr, inSizesAttr,
                                           padAttr, mpeModeAttr, clusterIdAttr);
            }
        };

        if (perShapeWorkloads.size() == 1) {
            // Single unique shape — check if dynamic channel split is needed
            const auto& sw = perShapeWorkloads[0];
            if (VPU::isDepthwiseOp(nceOp.getOperation()) &&
                ndType.getShape()[Dims4D::Act::C] == mlir::ShapedType::kDynamic && !sw.supportedChannels.empty()) {
                // Capture runtime channel count for dynamic materialization
                mlir::OpBuilder preBuilder(&ctx);
                preBuilder.setInsertionPoint(nceOp);
                auto input = nceOp->getOperand(0);
                auto channelDimIdx =
                        mlir::arith::ConstantIndexOp::create(preBuilder, nceOp->getLoc(), Dims4D::Act::C.ind());
                auto dynChannelCount = mlir::tensor::DimOp::create(preBuilder, nceOp->getLoc(), input, channelDimIdx);
                VPU::materializeWorkloadsDynamic(nceOp, builder, sw.descriptors, dynChannelCount, sw.supportedChannels,
                                                 _log);
            } else {
                VPU::materializeWorkloads(nceOp, builder, sw.descriptors);
            }
            // computeWorkloadsForShape restores/removes DPUCost internally (to avoid
            // side-effects from multi-shape calls), so without this the final IR would
            // have no DPUCost even though workloads were generated.
            nceOp->setAttr(DPUCost, getIntAttr(&ctx, sw.dpuCost));
            return;
        }

        // Multiple unique tile shapes — emit conditional materialization.
        // Dispatch on whichever operand extents actually tell the enumerated tiles apart
        // (selectDispatchKeys): a multi-operand op (e.g. a VF Eltwise skip connection) may need
        // a different operand's dynamic dim than the activation's.
        mlir::OpBuilder preBuilder(&ctx);
        preBuilder.setInsertionPoint(nceOp);

        auto dispatchKeys = selectDispatchKeys(enumeration.dispatchExtents, perShapeWorkloads.size());
        VPUX_THROW_WHEN(dispatchKeys.empty(),
                        "Cannot materialize multi-shape workloads for op at '{0}': no combination of the "
                        "dynamic operand dims tells the {1} enumerated tile shapes apart",
                        nceOp->getLoc(), perShapeWorkloads.size());

        // One tensor.dim per selected key, read once ahead of the per-tile conditions below.
        SmallVector<mlir::Value> runtimeDimSizes;
        for (auto keyIdx : dispatchKeys) {
            const auto& extent = enumeration.dispatchExtents[keyIdx];
            auto dimIdx = mlir::arith::ConstantIndexOp::create(preBuilder, nceOp->getLoc(), extent.dim);
            runtimeDimSizes.push_back(mlir::tensor::DimOp::create(preBuilder, nceOp->getLoc(),
                                                                  nceOp->getOperand(extent.operandIdx), dimIdx));
        }

        // Pre-create comparison conditions for each tile (AND over the selected dispatch keys).
        SmallVector<mlir::Value> conditions;
        for (size_t i = 0; i + 1 < perShapeWorkloads.size(); ++i) {
            auto dimCmp = [&](size_t d) {
                auto expectedSize = enumeration.dispatchExtents[dispatchKeys[d]].tileSizes[i];
                auto sizeConst = mlir::arith::ConstantIndexOp::create(preBuilder, nceOp->getLoc(), expectedSize);
                return mlir::arith::CmpIOp::create(preBuilder, nceOp->getLoc(), mlir::arith::CmpIPredicate::eq,
                                                   runtimeDimSizes[d], sizeConst);
            };

            mlir::Value cond = dimCmp(0);
            for (size_t d = 1; d < dispatchKeys.size(); ++d) {
                cond = mlir::arith::AndIOp::create(preBuilder, nceOp->getLoc(), cond, dimCmp(d));
            }
            conditions.push_back(cond);
        }

        auto& workloadRegion = nceOp.getWorkloads();
        if (workloadRegion.empty()) {
            workloadRegion.emplaceBlock();
        }

        builder.setInsertionPointToEnd(&workloadRegion.front());

        // Build nested scf.if chain inside the workloads region:
        //   if (cond[0]) { wl0 } else if (cond[1]) { wl1 } ... else { wlN }
        // Each else-region builder is used exactly once, so the chain is a plain loop.
        auto curBuilder = builder;
        for (size_t idx = 0; idx + 1 < perShapeWorkloads.size(); ++idx) {
            auto ifOp = mlir::scf::IfOp::create(curBuilder, nceOp->getLoc(), conditions[idx],
                                                /*withElseRegion=*/true);
            auto thenBuilder = ifOp.getThenBodyBuilder();
            emitWorkloads(thenBuilder, perShapeWorkloads[idx].descriptors);
            curBuilder = ifOp.getElseBodyBuilder();
        }
        emitWorkloads(curBuilder, perShapeWorkloads.back().descriptors);

        // Set a deterministic DPUCost attribute on the NCE op. Use the max cost across
        // all enumerated shapes (conservative for further scheduling).
        int64_t maxDpuCost = 0;
        for (const auto& sw : perShapeWorkloads) {
            maxDpuCost = std::max(maxDpuCost, sw.dpuCost);
        }
        nceOp->setAttr(DPUCost, getIntAttr(&ctx, maxDpuCost));
    });
}

}  // namespace

//
// createWorkloadsForNCEOpsSCFPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createWorkloadsForNCEOpsSCFPass(Logger log) {
    return std::make_unique<WorkloadsForNCEOpsSCFPass>(log);
}
