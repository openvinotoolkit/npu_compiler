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
#include "vpux/compiler/dialect/VPU/utils/workload_split_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/scope_exit.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

#include <vpu_cost_model.h>

#include <algorithm>
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

struct InsertionPath {
    mlir::OffsetSizeAndStrideOpInterface insertionOp;
    SmallVector<VPU::TilingBuilderOpInterface> shapeTransforms;
};

/**
 * Find the insertion op (OffsetSizeAndStrideOpInterface) that consumes the NCE result.
 * Records shape-changing operations that must be back-inferred from the inserted tile.
 * Returns std::nullopt if not found.
 *
 * Uses DestinationStyleOpInterface to identify insert-like ops (covers tensor.insert_slice).
 * tensor.parallel_insert_slice is also an insertion op but lives in the forall terminator
 * region and does not implement DestinationStyleOpInterface.
 */
std::optional<InsertionPath> findInsertionPath(VPU::NCEOpInterface nceOp) {
    // If the NCE result has multiple users, the insertion op is ambiguous (could lead to
    // non-deterministic shape enumeration). Fall back to bounded-shape handling.
    if (!nceOp->getResult(0).hasOneUse()) {
        return std::nullopt;
    }
    auto isInsertionOp = [](mlir::Operation* op) {
        return mlir::isa<mlir::DestinationStyleOpInterface>(op) || mlir::isa<mlir::tensor::ParallelInsertSliceOp>(op);
    };
    auto checkUser = [&](mlir::Operation* op) -> mlir::OffsetSizeAndStrideOpInterface {
        if (auto iface = mlir::dyn_cast<mlir::OffsetSizeAndStrideOpInterface>(op); iface && isInsertionOp(op)) {
            return iface;
        }
        return nullptr;
    };

    SmallVector<VPU::TilingBuilderOpInterface> shapeTransforms;

    // Follow the single-use chain: NCE -> [pass-through | shape-transform]* -> insertion_op, where
    //   pass-through   = tensor.cast | VPU.Copy | VPU.NCEEltwise  (shape-preserving, skipped silently)
    //   shape-transform = VPU.DepthToSpace                        (recorded in shapeTransforms so the
    //                                                              NCE tile shape can be back-inferred
    //                                                              from the inserted slice)
    // Handles patterns like:
    //   NCE -> insert_slice
    //   NCE -> tensor.cast -> insert_slice
    //   NCE -> VPU.Copy -> insert_slice
    //   NCE -> VPU.NCEEltwise -> insert_slice
    //   NCE -> tensor.cast -> VPU.Copy -> insert_slice
    //   NCE -> VPU.Copy -> tensor.cast -> insert_slice (unlikely but handled)
    //   NCE -> VPU.DepthToSpace -> insert_slice
    // Any other user aborts the walk. The depth cap bounds the search; longer chains fall back to
    // bounded-shape handling.
    mlir::Operation* cur = nceOp.getOperation();
    constexpr int maxChainDepth = 4;
    for (int depth = 0; depth < maxChainDepth; ++depth) {
        // Get the single user of the current op's result(0).
        mlir::Value result = cur->getResult(0);
        if (!result.hasOneUse()) {
            return std::nullopt;
        }
        mlir::Operation* user = *result.getUsers().begin();

        // Check if this user is the insertion op.
        if (auto iface = checkUser(user)) {
            return InsertionPath{iface, std::move(shapeTransforms)};
        }

        // Shape-preserving user: walk through it without recording a transform. VPU.NCEEltwise
        // qualifies because its output shape matches the operand shape.
        if (mlir::isa<mlir::tensor::CastOp, VPU::CopyOp, VPU::NCEEltwiseOp>(user)) {
            cur = user;
            continue;
        }

        if (auto depthToSpace = mlir::dyn_cast<VPU::DepthToSpaceOp>(user)) {
            shapeTransforms.push_back(mlir::cast<VPU::TilingBuilderOpInterface>(depthToSpace.getOperation()));
            cur = user;
            continue;
        }

        return std::nullopt;
    }
    return std::nullopt;
}

/**
 * Per-iteration tile shape pair: the NCE op's (output, input) shapes for one iteration.
 *
 * Border-tile distinction: two iterations may produce the same
 * output shape while their input slice differs (e.g. tile at the H/W boundary vs. an
 * interior tile when boundary padding is materialized via tensor.pad on the input).
 * Such tiles must be treated as distinct: their workload cost — and thus the split
 * decision — depends on the input size, not only the output size. Mirrors what
 * `getUniqueShapeTilingCandidates` does in the default (non-SCF) tiling path.
 */
struct TileShapes {
    Shape outputShape;
    Shape inputShape;
};

// Resolve a bounded tensor to a concrete shape for workload calculation. Bounds are substitutes only for dynamic
// dimensions; a static dimension remains authoritative even when its carried bound is stale or less precise. For
// example, tensor<1x?x11x16, bounds=[1, 48, 64, 16]> resolves to [1, 48, 11, 16], not [1, 48, 64, 16]. Such types
// appear when SCF tiling makes a dimension static while preserving bounds propagated from the untiled tensor.
Shape getShapeUsingBoundsForDynamicDims(mlir::Type type) {
    const auto ndType = mlir::cast<vpux::NDTypeInterface>(type);
    const auto shape = ndType.getShape();
    auto resolvedShape = Shape(getBoundedShape(ndType));

    for (const auto dim : irange(shape.size())) {
        if (shape[Dim(dim)] != mlir::ShapedType::kDynamic) {
            resolvedShape[Dim(dim)] = shape[Dim(dim)];
        }
    }

    return resolvedShape;
}

/**
 * Find the OffsetSizeAndStrideOpInterface op (typically tensor.extract_slice) that
 * provides the activation slice feeding the NCE op. The lookup walks backwards through
 * optional tensor.cast / VPU.Copy ops and an optional tensor.pad in any order:
 *
 *   extract_slice -> [tensor.cast|VPU.Copy|tensor.pad]* -> NCE
 *
 * If a tensor.pad is encountered anywhere in the chain, it is returned via `outPadOp`
 * so the caller can account for low/high padding when computing the effective NCE input
 * shape per iteration.
 *
 * Returns nullptr if the producing op is not an OffsetSizeAndStrideOpInterface.
 */
mlir::OffsetSizeAndStrideOpInterface findInputExtractionOp(VPU::NCEOpInterface nceOp, mlir::tensor::PadOp& outPadOp) {
    outPadOp = nullptr;
    mlir::Value input = nceOp->getOperand(0);
    // Walk back through optional tensor.cast / VPU.Copy and an optional tensor.pad
    // to reach the extraction op (tensor.extract_slice).
    for (int depth = 0; depth < 4; ++depth) {
        auto* defOp = input.getDefiningOp();
        if (!defOp) {
            return nullptr;
        }
        if (mlir::isa<mlir::tensor::CastOp, VPU::CopyOp, VPU::ConvertOp>(defOp)) {
            input = defOp->getOperand(0);
            continue;
        }
        if (auto padOp = mlir::dyn_cast<mlir::tensor::PadOp>(defOp)) {
            outPadOp = padOp;
            input = padOp.getSource();
            continue;
        }
        if (auto iface = mlir::dyn_cast<mlir::OffsetSizeAndStrideOpInterface>(defOp)) {
            return iface;
        }
        return nullptr;
    }
    return nullptr;
}

/**
 * Enumerate the unique per-iteration (output, input) tile-shape pairs for an NCE op
 * that lives inside one or more scf.for / scf.forall loops, evaluating ALL dynamic
 * dimensions of the output via the insertion op and of the input via the extraction
 * op (plus optional tensor.pad).
 *
 * Walks the Cartesian product of all parent loop iterations and, for each iteration,
 * evaluates both the output slice sizes (from the insert_slice / parallel_insert_slice)
 * and the input slice sizes (from the extract_slice, augmented by tensor.pad low+high
 * when present) to derive full (output, input) shapes. Results are deduplicated by the
 * pair so that border tiles whose output matches a middle tile but whose input differs
 * are preserved as distinct entries.
 *
 * Returns an empty vector on any failure (cannot find loops / ops, OFR evaluation
 * fails, or iteration count would exceed the cap).
 *
 * TODO: E#222387 — Refactor to reuse the IV collection and iteration-space evaluation
 * infrastructure from scf_loop_analysis_and_debug / scf_analyzer (collectIVArgs,
 * constructIterationSpace, evaluateOpValuePerIteration). The scf_analyzer path needs
 * scf.forall support first.
 */
SmallVector<TileShapes> getUniqueDynamicTileShapes(VPU::NCEOpInterface nceOp, Logger log) {
    auto outputType = nceOp->getResult(0).getType();
    auto outBounded = mlir::dyn_cast<vpux::Core::BoundedTensorType>(outputType);
    if (!outBounded) {
        return {};
    }
    const auto outShapeView = mlir::cast<vpux::NDTypeInterface>(outputType).getShape();
    const auto outBaseShape = getShapeUsingBoundsForDynamicDims(outputType);

    auto inputType = nceOp->getOperand(0).getType();
    auto inNd = mlir::cast<vpux::NDTypeInterface>(inputType);
    const auto inShapeView = inNd.getShape();
    auto inBounded = mlir::dyn_cast<vpux::Core::BoundedTensorType>(inputType);
    // Baseline input shape uses bounds (static dims) and gets dynamic dims overridden per iteration.
    Shape inBaseShape = inBounded ? getShapeUsingBoundsForDynamicDims(inputType) : inNd.getShape().toValues();

    // Identify which dimensions are dynamic for the output and the input.
    SmallVector<int64_t> outDynDims;
    for (int64_t i = 0; i < static_cast<int64_t>(outShapeView.size()); ++i) {
        if (outShapeView[Dim(i)] == mlir::ShapedType::kDynamic) {
            outDynDims.push_back(i);
        }
    }
    SmallVector<int64_t> inDynDims;
    for (int64_t i = 0; i < static_cast<int64_t>(inShapeView.size()); ++i) {
        if (inShapeView[Dim(i)] == mlir::ShapedType::kDynamic) {
            inDynDims.push_back(i);
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

    mlir::tensor::PadOp inputPadOp;
    auto extractionOp = findInputExtractionOp(nceOp, inputPadOp);
    if (!extractionOp) {
        // No extract_slice on the activation (e.g., activation comes directly from a function
        // argument or another op). Per-iteration input shape cannot vary, so we fall back to
        // the bounded baseline input shape for every iteration. Border-tile distinction reduces
        // to the previous behavior of enumerating output shapes only.
        log.trace("No input extraction op found — falling back to bounded input shape (no per-iteration "
                  "input variation)");
    }

    auto insertionPath = findInsertionPath(nceOp);
    if (!insertionPath.has_value() && !extractionOp) {
        log.trace("No insertion or input extraction op found — cannot derive per-iteration output shapes");
        return {};
    }

    SmallVector<mlir::OpFoldResult> outMixedSizes;
    if (insertionPath.has_value()) {
        outMixedSizes = llvm::to_vector(insertionPath->insertionOp.getMixedSizes());
    }
    SmallVector<mlir::OpFoldResult> inMixedSizes;
    if (extractionOp) {
        inMixedSizes = llvm::to_vector(extractionOp.getMixedSizes());
    }

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

    // Evaluate a single OpFoldResult against a single-valued IV map; returns the scalar value.
    auto evalSingle = [&](mlir::OpFoldResult ofr, ValueRangeMap& m) -> std::optional<int64_t> {
        auto opt = opAnalysis.getOpFoldResultValue(ofr, m, OpChainAnalysis::MODE::ALL_VALUES);
        if (!opt.has_value() || opt->empty()) {
            return std::nullopt;
        }
        return opt->front();
    };

    // Dedupe by (outputShape, inputShape) using std::set on std::vector pairs (lexicographic order).
    std::set<std::pair<std::vector<int64_t>, std::vector<int64_t>>> tileSet;

    SmallVector<size_t> idx(ivOrder.size(), 0);
    while (true) {
        // Build a single-valued IV map for this iteration.
        ValueRangeMap singleMap;
        for (size_t i = 0; i < ivOrder.size(); ++i) {
            singleMap[ivOrder[i]] = {ivValuesOrder[i][idx[i]]};
        }

        // Evaluate input shape for the current iteration (extract_slice + optional tensor.pad).
        Shape inShape(inBaseShape);
        bool ok = true;
        if (extractionOp) {
            for (auto dimI : inDynDims) {
                if (dimI >= static_cast<int64_t>(inMixedSizes.size())) {
                    ok = false;
                    break;
                }
                auto v = evalSingle(inMixedSizes[dimI], singleMap);
                if (!v.has_value() || *v <= 0) {
                    ok = false;
                    break;
                }
                int64_t sz = *v;
                if (inputPadOp) {
                    // tensor.pad augments each dim of the extracted slice by low+high padding.
                    auto lowPads = inputPadOp.getMixedLowPad();
                    auto highPads = inputPadOp.getMixedHighPad();
                    if (dimI < static_cast<int64_t>(lowPads.size())) {
                        if (auto lv = evalSingle(lowPads[dimI], singleMap)) {
                            sz += *lv;
                        }
                    }
                    if (dimI < static_cast<int64_t>(highPads.size())) {
                        if (auto hv = evalSingle(highPads[dimI], singleMap)) {
                            sz += *hv;
                        }
                    }
                }
                inShape[Dim(dimI)] = sz;
            }
        }

        // Prefer the insertion tile because it also captures downstream shape transforms. Intermediate NCE ops may
        // not have an insertion path; derive their dynamic spatial output dimensions from the concrete activation
        // tile instead of using the full bounded tensor shape.
        Shape outShape(outBaseShape);
        if (ok && insertionPath.has_value()) {
            Shape insertedShape(outMixedSizes.size());
            for (const auto dimI : irange(outMixedSizes.size())) {
                auto v = evalSingle(outMixedSizes[dimI], singleMap);
                if (!v.has_value() || *v <= 0) {
                    ok = false;
                    break;
                }
                insertedShape[Dim(dimI)] = *v;
            }

            if (ok) {
                TileInfo outputTile(insertedShape);
                for (auto shapeTransform : insertionPath->shapeTransforms | reversed) {
                    auto inputTiling = shapeTransform.backInferTileInfo(outputTile, log.nest());
                    if (inputTiling.tiles.size() != 1) {
                        ok = false;
                        break;
                    }
                    outputTile = inputTiling.tiles.front();
                }
                if (ok) {
                    outShape = outputTile.shape;
                }
            }
        } else if (ok) {
            const auto kernel = nceOp.getKernelSizeVal();
            const auto strides = nceOp.getStridesVal();
            const auto padding = VPU::toPadInfo(nceOp.getPad());
            const std::array spatialDims = {std::tuple{Dims4D::Act::H, kernel[Dims4D::Kernel::Y.ind()],
                                                       strides[Dims4D::Strides::Y.ind()], padding.top, padding.bottom},
                                            std::tuple{Dims4D::Act::W, kernel[Dims4D::Kernel::X.ind()],
                                                       strides[Dims4D::Strides::X.ind()], padding.left, padding.right}};
            for (const auto& [dim, kernelSize, stride, padBefore, padAfter] : spatialDims) {
                if (outShapeView[dim] != mlir::ShapedType::kDynamic) {
                    continue;
                }
                outShape[dim] = (inShape[dim] + padBefore + padAfter - kernelSize) / stride + 1;
                if (outShape[dim] <= 0) {
                    ok = false;
                    break;
                }
            }
            if (llvm::any_of(outDynDims, [&](int64_t dim) {
                    return dim != Dims4D::Act::H.ind() && dim != Dims4D::Act::W.ind();
                })) {
                ok = false;
            }
        }

        if (ok) {
            std::vector<int64_t> outKey(outShape.raw().begin(), outShape.raw().end());
            std::vector<int64_t> inKey(inShape.raw().begin(), inShape.raw().end());
            tileSet.insert({std::move(outKey), std::move(inKey)});
        }

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

    if (tileSet.empty()) {
        log.trace("Failed to derive per-iteration tile shapes for dynamic NCE op at '{0}'", nceOp->getLoc());
        return {};
    }

    SmallVector<TileShapes> result;
    result.reserve(tileSet.size());
    for (const auto& p : tileSet) {
        result.push_back({Shape(SmallVector<int64_t>(p.first.begin(), p.first.end())),
                          Shape(SmallVector<int64_t>(p.second.begin(), p.second.end()))});
    }

    // Sort descending by output total elements (largest first); stable on input shape for ties.
    llvm::sort(result, [](const TileShapes& a, const TileShapes& b) {
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
        for (auto v : a.inputShape.raw()) {
            ia *= v;
        }
        for (auto v : b.inputShape.raw()) {
            ib *= v;
        }
        return ia > ib;
    });

    log.trace("Found {0} unique (output, input) tile shapes for op at '{1}'", result.size(), nceOp->getLoc());
    return result;
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
                                        const Shape& explicitInputShape, mlir::MLIRContext* ctx, config::ArchKind arch,
                                        int64_t numDPUs, VPUNN::VPUCostModel& costModel, Logger log) {
    ShapeWorkloads result;
    result.outputShape = staticShape;
    result.inputShape = explicitInputShape;

    // --- Compute input shape ---
    // Use the explicit per-tile input shape provided by the caller (this captures
    // border-vs-interior tile differences when tensor.pad augments the extract_slice).
    // For depthwise ops, override the channel dimension to match the output.
    const bool isDepthwise = VPU::isDepthwiseOp(nceOp.getOperation());
    const auto channelSize = staticShape[Dims4D::Act::C];

    Shape inputShape = explicitInputShape;
    if (isDepthwise && inputShape.size() == 4 && inputShape[Dims4D::Act::C] != channelSize) {
        inputShape[Dims4D::Act::C] = channelSize;
    }

    // Build WorkloadCostParams directly. Pad info and MC strategy (with their consistency checks /
    // scf.forall inference) are derived inside getWorkloadCostParamForSCF
    auto costParams =
            VPU::getWorkloadCostParamForSCF(nceOp, arch, numDPUs, ShapeRef(inputShape), ShapeRef(staticShape), log);

    // Compute workload descriptors
    // Uses the pre-built cost params and explicit output shape for MPE mode computation.
    // generateWorkloadDescriptors sets DPUCost on the op as a side-effect.
    auto originalDPUCostAttr = nceOp->getAttrOfType<mlir::IntegerAttr>(DPUCost);
    SmallVector<std::pair<unsigned, mlir::Type>> originalInputTypes;
    VPUX_SCOPE_EXIT {
        for (auto& [idx, origType] : originalInputTypes) {
            nceOp->getOperand(idx).setType(origType);
        }

        if (originalDPUCostAttr) {
            nceOp->setAttr(DPUCost, originalDPUCostAttr);
        } else {
            nceOp->removeAttr(DPUCost);
        }
    };

    int64_t dpuCost = 0;
    result.descriptors =
            VPU::computeWorkloadDescriptorsForSCF(nceOp, costParams, ShapeRef(staticShape), costModel, dpuCost, log);
    result.dpuCost = dpuCost;

    // Apply corrections and back-infer input workloads
    // These functions still read from the op's input types, so we temporarily set them
    // to the resolved static shapes. The output type is NOT mutated.
    for (unsigned i = 0; i < nceOp->getNumOperands(); ++i) {
        auto opInputType = nceOp->getOperand(i).getType();
        auto opInputNdType = mlir::dyn_cast<vpux::NDTypeInterface>(opInputType);
        if (!opInputNdType) {
            continue;
        }

        auto opInputBounded = mlir::dyn_cast<vpux::Core::BoundedTensorType>(opInputType);
        const bool hasDynamic = opInputNdType.getShape().isDynamic();
        const bool needsChannelOverride = isDepthwise && i == 0 && opInputNdType.getShape().size() >= 4 &&
                                          opInputNdType.getShape()[Dims4D::Act::C] != channelSize;

        if (!hasDynamic && !needsChannelOverride) {
            continue;
        }

        Shape inShape;
        if (i == 0) {
            // Use the explicit per-shape input tile shape so spatial back-inference clamps correctly.
            inShape = inputShape;
        } else if (opInputBounded) {
            inShape = Shape(opInputBounded.getBounds().raw());
        } else {
            inShape = opInputNdType.getShape().toValues();
        }
        if (needsChannelOverride) {
            inShape[Dims4D::Act::C] = channelSize;
        }
        auto operandVal = nceOp->getOperand(i);
        originalInputTypes.push_back({i, opInputType});
        operandVal.setType(vpux::getTensorType(ShapeRef(inShape), opInputNdType.getElementType(),
                                               opInputNdType.getDimsOrder(), opInputNdType.getMemSpace()));
    }

    VPU::correctWorkloadDescriptors(nceOp, result.descriptors, ctx, log, &result.supportedChannels);

    VPU::backInferInputWorkloads(nceOp, result.descriptors, log);

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

        auto inputType = nceOp->getOperand(0).getType();
        auto inputNdType = mlir::cast<vpux::NDTypeInterface>(inputType);
        const auto inputShapeView = inputNdType.getShape();

        // Determine the set of unique (output, input) tile shape pairs to compute workloads for.
        // Static output -> single pair; dynamic output -> enumerate per-iteration pairs (preserves
        // border tiles whose input slice differs from interior tiles with the same output shape).
        SmallVector<TileShapes> tileShapes;
        if (!outputShape.isDynamic()) {
            Shape inShape = mlir::isa<vpux::Core::BoundedTensorType>(inputType)
                                    ? getShapeUsingBoundsForDynamicDims(inputType)
                                    : inputNdType.getShape().toValues();
            tileShapes.push_back({outputShape.toValues(), std::move(inShape)});
        } else {
            auto boundedType = mlir::dyn_cast<vpux::Core::BoundedTensorType>(outputType);
            VPUX_THROW_WHEN(boundedType == nullptr, "NCE op at '{0}' has dynamic output without bounds",
                            nceOp->getLoc());

            // Enumerate per-iteration (output, input) tile shapes from parent scf.for / scf.forall
            // loops, accounting for tensor.pad on the input. Two iterations sharing an output
            // shape but differing on input are kept distinct.
            tileShapes = getUniqueDynamicTileShapes(nceOp, _log);

            if (tileShapes.empty()) {
                // Fallback: use bounded output shape and bounded/static input shape.
                // This handles cases where the NCE op's output doesn't flow into an
                // insert_slice (e.g., multi-step reductions where the first NCE feeds
                // a second reduction op rather than being inserted back into an accumulator).
                _log.trace("Falling back to bounded shapes for op at '{0}'", nceOp->getLoc());
                Shape outShape = getShapeUsingBoundsForDynamicDims(outputType);
                Shape inShape = mlir::isa<vpux::Core::BoundedTensorType>(inputType)
                                        ? getShapeUsingBoundsForDynamicDims(inputType)
                                        : inputNdType.getShape().toValues();
                tileShapes.push_back({std::move(outShape), std::move(inShape)});
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
            _log.trace("Computing workloads for tile shape [{0}/{1}]: outputShape={2}, inputShape={3}", shapeIdx + 1,
                       tileShapes.size(), ts.outputShape, ts.inputShape);
            auto sw = computeWorkloadsForShape(nceOp, ts.outputShape, ts.inputShape, &ctx, arch, numDPUs, *costModel,
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
        // Dispatch on the NCE op's INPUT dynamic dims via tensor.dim: input is fully
        // accessible before the NCE op, and each tile is uniquely identified by its
        // input shape (border-vs-interior distinction is captured in the input shape
        // when tensor.pad augments the extract_slice).
        mlir::OpBuilder preBuilder(&ctx);
        preBuilder.setInsertionPoint(nceOp);
        auto input = nceOp->getOperand(0);

        // Capture runtime sizes for every dynamic dim of the NCE input.
        // Note: multi-shape materialization requires at least one dynamic input dim to dispatch on.
        SmallVector<int64_t> dispatchDimIndices;
        SmallVector<mlir::Value> runtimeDimSizes;

        for (int64_t i = 0; i < static_cast<int64_t>(inputShapeView.size()); ++i) {
            if (inputShapeView[Dim(i)] != mlir::ShapedType::kDynamic) {
                continue;
            }
            dispatchDimIndices.push_back(i);
            auto dimIdx = mlir::arith::ConstantIndexOp::create(preBuilder, nceOp->getLoc(), i);
            auto dynSize = mlir::tensor::DimOp::create(preBuilder, nceOp->getLoc(), input, dimIdx);
            runtimeDimSizes.push_back(dynSize);
        }

        VPUX_THROW_WHEN(dispatchDimIndices.empty(),
                        "Cannot materialize multi-shape workloads for op at '{0}': input has no "
                        "dynamic dimensions to dispatch on (output dimensions are not available "
                        "before the op is defined)",
                        nceOp->getLoc());

        // Pre-create comparison conditions for each tile (AND over dispatch dynamic dims).
        SmallVector<mlir::Value> conditions;
        for (size_t i = 0; i + 1 < perShapeWorkloads.size(); ++i) {
            const auto& shape = perShapeWorkloads[i].inputShape;
            auto dimCmp = [&](size_t d) {
                auto expectedSize = shape[Dim(dispatchDimIndices[d])];
                auto sizeConst = mlir::arith::ConstantIndexOp::create(preBuilder, nceOp->getLoc(), expectedSize);
                return mlir::arith::CmpIOp::create(preBuilder, nceOp->getLoc(), mlir::arith::CmpIPredicate::eq,
                                                   runtimeDimSizes[d], sizeConst);
            };

            mlir::Value cond = dimCmp(0);
            for (size_t d = 1; d < dispatchDimIndices.size(); ++d) {
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

        // Set a deterministic DPUCost attribute on the NCE op. computeWorkloadsForShape
        // restores/removes DPUCost internally (to avoid side-effects from multi-shape calls),
        // so without this the final IR would have no DPUCost even though workloads were generated.
        // Use the max cost across all enumerated shapes (conservative for downstream scheduling).
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
