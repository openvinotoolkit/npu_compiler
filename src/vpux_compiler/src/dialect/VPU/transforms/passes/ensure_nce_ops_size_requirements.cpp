//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/sparsity.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Transforms/DialectConversion.h>

#include <numeric>

namespace vpux::VPU {
#define GEN_PASS_DECL_ENSURENCEOPSSIZEREQUIREMENTS
#define GEN_PASS_DEF_ENSURENCEOPSSIZEREQUIREMENTS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

constexpr llvm::StringLiteral preserveConcatAlignedICSplitAttr = "preserve_concat_aligned_ic_split";

SmallVector<Dim> getDimsOverKHWLimit(ShapeRef shape, ArrayRef<int64_t> dimThresholds) {
    SmallVector<Dim> wrongDims = {};
    for (size_t i = 0; i < shape.size(); i++) {
        const auto dim = Dim(i);
        if (shape[dim] > dimThresholds[i]) {
            wrongDims.push_back(dim);
        }
    }
    return wrongDims;
}

bool hasSplitOverKernelStrategy(mlir::Operation* op) {
    if (mlir::isa<VPU::ClusteredOpInterface>(op)) {
        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(op);
        const auto strategy = clusteredOp.getMultiClusterStrategy();
        if (!strategy.has_value()) {
            return false;
        }

        return strategy.value() == VPU::MultiClusterStrategy::SplitOverKernel;
    }

    return false;
}

static int64_t combineOptionalAlignments(int64_t lhs, int64_t rhs) {
    if (lhs == 0) {
        return rhs;
    }

    if (rhs == 0) {
        return lhs;
    }

    return std::lcm(lhs, rhs);
}

static int64_t getAlignmentFromDynamicDequantize(mlir::Operation* op, bool enableSplitChannelForDynamicDequantize) {
    if (!enableSplitChannelForDynamicDequantize) {
        return 0;
    }

    auto nceConvOp = mlir::dyn_cast_if_present<VPU::NCEConvolutionOp>(op);
    if (nceConvOp == nullptr) {
        return 0;
    }
    auto weights = nceConvOp.getFilter();

    // Walk through view-like ops
    auto currentOp = weights.getDefiningOp();
    while (mlir::isa_and_present<VPU::ViewLikeOpInterface>(currentOp) && currentOp->hasOneUse()) {
        currentOp = currentOp->getOperand(0).getDefiningOp();
    }
    if (!mlir::isa_and_present<VPU::DynamicDequantizeOp>(currentOp) || !currentOp->hasOneUse()) {
        return 0;
    }

    auto filterShape = getShape(weights);
    auto dequantInShape = getShape(currentOp->getOperand(0));
    if (filterShape.size() != 4 || dequantInShape.size() != 4) {
        return 0;
    }

    bool mergedIC =
            filterShape[Dims4D::Filter::OC] == dequantInShape[Dims4D::Act::C] &&
            filterShape[Dims4D::Filter::IC] == dequantInShape[Dims4D::Act::H] * dequantInShape[Dims4D::Act::W] &&
            filterShape[Dims4D::Filter::KY] == 1 && filterShape[Dims4D::Filter::KX] == 1;
    if (!mergedIC) {
        return 0;
    }

    // The post AffineReshape of DynamicDequantize merges [H, W] into IC, so alignment is set to W for
    // making pattern SplitMemPermuteAndDynamicDequantizeOp work
    return dequantInShape[Dims4D::Act::W];
}

// Matches the subgraph pattern for NCEConvolutionOp:
//   weights <- Slice <- PermuteCast <- AffineReshape (single use) <- Concat (single use,
//   >= 3 inputs, single axis, arg0..argN-3 equal-sized on that axis).
// Returns the LCM of the per-input concat-axis size and the NCE channel alignment,
// or 0 if the pattern does not match.
static int64_t getAlignmentFromConcatSlicePattern(mlir::Operation* op) {
    auto nceConvOp = mlir::dyn_cast_if_present<VPU::NCEConvolutionOp>(op);
    if (nceConvOp == nullptr) {
        return 0;
    }
    auto weights = nceConvOp.getFilter();

    // Trace from weights
    auto sliceOp = weights.getDefiningOp<VPU::SliceOp>();
    if (sliceOp == nullptr) {
        return 0;
    }
    auto permuteCastOp = sliceOp.getInput().getDefiningOp<VPU::PermuteCastOp>();
    if (permuteCastOp == nullptr) {
        return 0;
    }
    auto affineReshapeOp = permuteCastOp.getInput().getDefiningOp<VPU::AffineReshapeOp>();
    if (affineReshapeOp == nullptr || !affineReshapeOp->hasOneUse()) {
        return 0;
    }
    auto concatOp = affineReshapeOp.getInput().getDefiningOp<VPU::ConcatOp>();
    if (concatOp == nullptr || !concatOp->hasOneUse()) {
        return 0;
    }
    // Require exactly one concat axis.
    const auto concatAxes = VPU::getConcatAxes(concatOp);
    if (concatAxes.size() != 1) {
        return 0;
    }
    const auto axis = Dim(*concatAxes.begin());
    const auto inputs = concatOp.getInputs();
    const int64_t firstSize = getShape(inputs.front())[axis];
    if (inputs.size() > 2) {
        // inputs[0] through inputs[N-3] must have equal size on the concat axis.
        // The last two inputs are allowed to differ (e.g. remainder tiles).
        for (auto input : inputs.drop_back(2)) {
            if (getShape(input)[axis] != firstSize) {
                return 0;
            }
        }
    }

    // Align to the LCM of the per-input concat size and the NCE channel alignment
    // so that each tile satisfies both the concat boundary and HW alignment requirements.
    auto weightsType = mlir::cast<vpux::NDTypeInterface>(weights.getType());
    const int64_t nceAlignment = VPU::NCEInvariant::getAlignment(weightsType.getElementType());
    const int64_t alignment = std::lcm(firstSize, nceAlignment);

    // If the alignment exceeds the HW dimension limit, tiles cannot be produced within
    // that limit, so fall back to the default alignment.
    if (alignment > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        return 0;
    }
    return alignment;
}

class EnsureNCEOpSizeRequirements final : public mlir::OpInterfaceRewritePattern<VPU::TilingBuilderOpInterface> {
public:
    EnsureNCEOpSizeRequirements(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::TilingBuilderOpInterface>(ctx), _log(log) {
        this->setDebugName("EnsureNCEOpSizeRequirements");
    }
    mlir::LogicalResult matchAndRewrite(VPU::TilingBuilderOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EnsureNCEOpSizeRequirements::matchAndRewrite(VPU::TilingBuilderOpInterface origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    auto op = origOp.getOperation();
    // Skip ops with fused reduce outputs — tiling does not support multi-result NCE ops yet.
    // TODO: E#209747 implement reduce output support after tiling, and remove this check
    if (mlir::isa<VPU::NCEOpInterface>(op) && op->getNumResults() > 1) {
        _log.trace("[{0}] Skipping: multi-result NCE op at '{1}'", this->getDebugName(), origOp->getLoc());
        return mlir::failure();
    }

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());
    rewriter.setInsertionPoint(op);

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    const auto outputShape = outputType.getShape();
    Shape nTilesOnDim(outputShape.size(), 1);
    const auto log = _log.nest();
    const auto tilingMode = TilingMode::ISOLATED;
    const auto tileDimOrder = getTileDimOrder(op, tilingMode, log);
    _log.nest(4).trace("Tile Dim order is {0}", tileDimOrder);
    const auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
    const auto numClusters = config::getTileExecutor(moduleOp).getCount();
    const int64_t concatInputAlignment = getAlignmentFromConcatSlicePattern(op);

    const auto getTilesWithOptionalConcatAlignment = [&](ShapeRef tilesOnDim) -> mlir::FailureOr<OutputTiling> {
        if (concatInputAlignment == 0) {
            return fillDividedTiles(op, tilesOnDim, outputShape);
        }

        // Preserve op-specific alignment and tile-unroll order, then LCM the C-dimension
        // alignment with the concat input alignment to respect concat boundaries.
        auto alignment = getAlignment(op, tilesOnDim, outputShape);
        alignment[Dims4D::Act::C.ind()] = std::lcm(alignment[Dims4D::Act::C.ind()], concatInputAlignment);
        auto optionalAlignment = std::optional<ArrayRef<int64_t>>(alignment);
        auto unrollSpatialFirst = isSpatialFirstNestedTiling(op, tilesOnDim);
        return fillDividedTiles(tilesOnDim, outputShape, optionalAlignment, unrollSpatialFirst);
    };

    const auto isSupportedTileSize = [&](ShapeRef nTilesOnDim, Dim dimToTile, ArrayRef<int64_t> dimThresholds) -> bool {
        const auto tiles = getTilesWithOptionalConcatAlignment(nTilesOnDim);
        if (mlir::failed(tiles)) {
            return false;
        }
        for (auto tile : tiles.value()) {
            if (tile.shape[dimToTile] > dimThresholds[dimToTile.ind()]) {
                return false;
            }
            auto inputTiling = origOp.backInferTileInfo(tile, log);
            auto& inTiles = inputTiling.tiles;
            if ((dimToTile != Dims4D::Act::C) &&
                (inTiles.begin()->shape[dimToTile] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT)) {
                return false;
            }
        }
        return true;
    };

    // Construct dim-specific thresholds for input and output shapes
    // In our test, extending the threshold on Dim C can improve performance by reducing workloads for SOK NCE
    // operations when the number of clusters is greater than 2
    SmallVector<int64_t> outputDimThresholds(outputShape.size(), VPU::NCEInvariant::VPU_DIMENSION_LIMIT);
    if (hasSplitOverKernelStrategy(op) && numClusters > 2) {
        outputDimThresholds[(Dims4D::Act::C).ind()] = VPU::NCEInvariant::VPU_DIMENSION_LIMIT * numClusters;
    }

    for (auto tileDimIter = tileDimOrder.begin(); tileDimIter < tileDimOrder.end(); ++tileDimIter) {
        auto dimToTile = *tileDimIter;
        while (!isSupportedTileSize(nTilesOnDim, dimToTile, outputDimThresholds) &&
               nTilesOnDim[dimToTile] <= outputShape[dimToTile]) {
            _log.nest(1).trace("Failed to tile {0} at {1} with {2}", op->getName(), dimToTile, nTilesOnDim);
            ++nTilesOnDim[dimToTile];
        }
    }

    // In case of single tile scheduled there is no need for tiling
    if (llvm::none_of(nTilesOnDim, [](int64_t tiles) {
            return tiles > 1;
        })) {
        return mlir::failure();
    }

    const auto tilesNew = getTilesWithOptionalConcatAlignment(nTilesOnDim);
    if (mlir::failed(tilesNew)) {
        return mlir::failure();
    }

    _log.nest(1).trace("Apply Tiling Strategy for {0} with {1}", op->getName(), nTilesOnDim);
    return VPU::applyTileStrategy(origOp, tilesNew.value(), rewriter, log.nest());
}

//
//  EnsureConvICRequirements helpers
//

static bool shouldSkipICSplitDueToConcatAlignment(VPU::NCEConvolutionOp convOp, int64_t maxTileIC) {
    if (convOp->hasAttr(preserveConcatAlignedICSplitAttr)) {
        return true;
    }

    const int64_t icAlign = getAlignmentFromConcatSlicePattern(convOp);
    const int64_t inputC = getShape(convOp.getInput())[Dims4D::Act::C];
    return icAlign > 0 && icAlign > maxTileIC && inputC <= VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
}

// Heuristic: estimate the largest IC per tile that fits in CMX under a double-buffered
// (pipelined) memory layout using minimum-efficient tile dimensions (H/W=8, OC=64).
// This gives an optimistic upper bound; tiles with IC above this threshold are split further.
//
// Two pipeline layouts are considered:
//   (A) 2 × input + 1 × weight + 2 × output  (double-buffer activation)
//   (B) 1 × input + 2 × weight + 2 × output  (double-buffer weights)
//
// The workload is considered efficient only when H >= 8, W >= 8, OC >= 64.
// When the actual dimension is smaller than the threshold, the actual dimension is used instead.
// On non-supported architectures the function returns VPU_DIMENSION_LIMIT
// so the caller falls back to the legacy behaviour.
//
// The returned value is already floored to the IC alignment required by the weight layout.
//
// TODO (E#208499): this helper is a transitional step. The pipelining-aware
// IC split decision will be folded into the general tiling search range later.
static int64_t computeTileBytes(int64_t elementCount, mlir::Type elemType) {
    const int64_t totalBits = elementCount * vpux::getElemTypeSize(elemType).count();
    return llvm::divideCeil(totalBits, int64_t{8});
}

static int64_t computeMaxPipelinableIC(VPU::NCEConvolutionOp op) {
    // Apply the dynamic pipelining-aware IC threshold only on supported architectures.
    // Older architectures (e.g. NPU40XX) fall back to the legacy VPU_DIMENSION_LIMIT.
    if (!VPU::isPipelineAwareConvSplitOverICSupported(op)) {
        return VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
    }

    constexpr int64_t minH = 8;
    constexpr int64_t minW = 8;
    constexpr int64_t minOC = 64;

    const auto inputShape = getShape(op.getInput());
    const auto outType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    const auto outputShape = outType.getShape();

    const auto inType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    const auto weightsType = mlir::cast<vpux::NDTypeInterface>(op.getFilter().getType());
    const auto kernelShape = getShape(op.getFilter());
    const auto kernelH = kernelShape[Dims4D::Filter::KY];
    const auto kernelW = kernelShape[Dims4D::Filter::KX];

    // IC alignment derived from data types (matches the existing tiling alignment logic).
    const auto inAlignment = VPU::NCEInvariant::getAlignment(inType.getElementType());
    const auto weightsAlignment = VPU::NCEInvariant::getAlignment(weightsType.getElementType());
    const int64_t icAlignment =
            ((inAlignment * kernelW * kernelH) % weightsAlignment == 0) ? inAlignment : weightsAlignment;

    // Per-IC-slice byte footprints use the minimum efficient tile dimensions (minH, minW, minOC)
    // rather than the actual op dimensions. This gives the largest admissible maxTileIC: we are
    // asking "how many IC can fit when the tile is the smallest workload still considered
    // efficient?" so that any tile meeting the H/W/OC guard above can be pipelined.
    // If the actual dimension is smaller than the minimum threshold, use the actual dimension.
    // (round bits up to whole bytes).
    const int64_t tileH = std::min(inputShape[Dims4D::Act::H], minH);
    const int64_t tileW = std::min(inputShape[Dims4D::Act::W], minW);
    const int64_t tileOC = std::min(outputShape[Dims4D::Act::C], minOC);

    const auto strides = parseIntArrayAttr<int64_t>(op.getStrides());
    const int64_t strideH = strides[0];
    const int64_t strideW = strides[1];
    const int64_t outTileH = (tileH + strideH - 1) / strideH;
    const int64_t outTileW = (tileW + strideW - 1) / strideW;

    const int64_t inBytesPerIC = computeTileBytes(inputShape[Dims4D::Act::N] * tileH * tileW, inType.getElementType());
    const int64_t weightBytesPerIC = computeTileBytes(tileOC * kernelH * kernelW, weightsType.getElementType());
    const int64_t outBytes =
            computeTileBytes(outputShape[Dims4D::Act::N] * tileOC * outTileH * outTileW, outType.getElementType());

    const int64_t cmxBytes = VPU::getTotalCMXSize(op).count();
    const int64_t cmxForTiles = cmxBytes - 2 * outBytes;

    if (cmxForTiles <= 0) {
        // Output alone already exceeds CMX; fall back to hardware limit.
        return VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
    }

    // Choose the less restrictive pipelining pattern:
    // max(cmxForTiles/(2*in+weight), cmxForTiles/(in+2*weight)) = cmxForTiles/min(2*in+weight, in+2*weight)
    // where min(2*in+weight, in+2*weight) = in + weight + min(in, weight).
    int64_t maxTileIC = 0;
    if (const int64_t denomMin = inBytesPerIC + weightBytesPerIC + std::min(inBytesPerIC, weightBytesPerIC);
        denomMin > 0) {
        maxTileIC = cmxForTiles / denomMin;
    }
    VPUX_THROW_UNLESS(icAlignment > 0, "IC alignment must be positive");
    maxTileIC = (maxTileIC / icAlignment) * icAlignment;

    if (maxTileIC <= 0) {
        return VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
    }

    // Never exceed the hardware dimension limit.
    return std::min(maxTileIC, VPU::NCEInvariant::VPU_DIMENSION_LIMIT);
}

//
//  EnsureConvICRequirements
//

class EnsureConvICRequirements final : public mlir::OpRewritePattern<VPU::NCEConvolutionOp> {
public:
    EnsureConvICRequirements(mlir::MLIRContext* ctx, bool enableSplitChannelForDynamicDequantize, Logger log)
            : mlir::OpRewritePattern<VPU::NCEConvolutionOp>(ctx),
              _enableSplitChannelForDynamicDequantize(enableSplitChannelForDynamicDequantize),
              _log(log) {
        this->setDebugName("EnsureConvICRequirements");
    }
    mlir::LogicalResult matchAndRewrite(VPU::NCEConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableSplitChannelForDynamicDequantize;
    Logger _log;
};

mlir::LogicalResult EnsureConvICRequirements::matchAndRewrite(VPU::NCEConvolutionOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    // Skip if origOp has reduce outputs, which are not supported after the split.
    // TODO: E#209747 implement reduce output support after split, and remove this check
    if (VPU::hasReduceOutputs(origOp)) {
        _log.trace("[{0}] Skipping NCEConvolutionSplitOverInputChannel: NCEConvolution with reduce outputs at '{1}'",
                   this->getDebugName(), origOp->getLoc());
        return mlir::failure();
    }

    // Split over IC supported only for NCEConvolutionOp
    // TODO: E#70421

    const auto inputShape = getShape(origOp.getInput());
    const auto inputC = inputShape[Dims4D::Act::C];

    // Compute the maximum IC per tile that still allows CMX pipelining with sufficient
    // spatial workload size for good DPU efficiency (H/W >= 8, OC >= 64).
    const int64_t maxTileIC = computeMaxPipelinableIC(origOp);

    _log.trace("[{0}] op '{1}' inputShape={2} inputC={3} maxTileIC={4}", this->getDebugName(), origOp->getLoc(),
               inputShape, inputC, maxTileIC);

    if (inputC <= maxTileIC) {
        return mlir::failure();
    }

    if (shouldSkipICSplitDueToConcatAlignment(origOp, maxTileIC)) {
        _log.trace("[{0}] op '{1}' skipping IC split due to concat alignment constraint", this->getDebugName(),
                   origOp->getLoc());
        return mlir::failure();
    }

    const auto kernelShape = getShape(origOp.getFilter());
    const auto kernelW = kernelShape[Dims4D::Filter::KX];
    const auto kernelH = kernelShape[Dims4D::Filter::KY];

    const int64_t icAlign = combineOptionalAlignments(
            getAlignmentFromConcatSlicePattern(origOp),
            getAlignmentFromDynamicDequantize(origOp, _enableSplitChannelForDynamicDequantize));
    const int64_t effectiveMaxTileIC = (icAlign > maxTileIC) ? icAlign : maxTileIC;

    if (effectiveMaxTileIC == 0) {
        _log.trace("[{0}] Invalid effectiveMaxTileIC=0 at '{1}'", this->getDebugName(), origOp->getLoc());
        return mlir::failure();
    }

    const auto maxTiles = vpux::divUp(inputC, effectiveMaxTileIC);

    if (maxTiles == 1) {
        return mlir::failure();
    }

    Shape nTilesOnDim(inputShape.size(), 1);
    nTilesOnDim[Dims4D::Act::C] = maxTiles;

    SmallVector<int64_t> alignment(inputShape.size(), 1);
    auto weightsType = mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType());
    auto alignedOp = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(origOp.getOperation());
    const auto inAlignment = alignedOp.getInputChannelAlignment();
    const auto weightsAlignment =
            VPU::NCEInvariant::getWeightSetAlignment(origOp.getOperation(), weightsType.getElementType());

    // Weights alignment requirement is IC * KH * KW aligned with weightsAlignment. For
    // int4 case, weightsAlignment = 32, if KH = 2, then IC = 16 can meet the requirement.
    // So here we first check if inAlignment can meet the requirement or not.
    const auto kernelSpatialSize = kernelH * kernelW;
    const auto weightsICAlignmentDivisor = std::gcd(weightsAlignment, kernelSpatialSize);
    if (weightsICAlignmentDivisor == 0) {
        return mlir::failure();
    }
    const auto weightsICAlignment = weightsAlignment / weightsICAlignmentDivisor;

    // The IC tile must satisfy both input activation alignment and weight-set alignment.
    alignment[Dims4D::Act::C.ind()] = std::lcm(inAlignment, weightsICAlignment);

    if (icAlign != 0) {
        alignment[Dims4D::Act::C.ind()] = std::lcm(alignment[Dims4D::Act::C.ind()], icAlign);
    }

    auto optionalAlignment = std::optional<ArrayRef<int64_t>>(alignment);
    _log.trace("[{0}] op '{1}' nTilesOnDim={2} alignment={3} icAlign={4}", this->getDebugName(), origOp->getLoc(),
               nTilesOnDim, alignment, icAlign);
    const auto tiles = fillDividedTiles(nTilesOnDim, inputShape, optionalAlignment);

    if (mlir::failed(tiles)) {
        return mlir::failure();
    }

    auto weightInput = origOp.getFilter();
    // check for parent weight shave dequantize op
    auto weightDequantizeOp = weightInput.getDefiningOp<VPU::DequantizeOp>();
    if (weightDequantizeOp != nullptr) {
        weightInput = weightDequantizeOp.getInput();
    }

    SmallVector<VPU::NCEConvolutionOp> convOps;
    SmallVector<VPU::NCEEltwiseOp> addOps;
    SmallVector<VPU::DequantizeOp> dequantizeOps;
    mlir::Value result = VPU::splitNCEConvolutionOverIC(origOp, weightInput, convOps, addOps, dequantizeOps,
                                                        tiles.value(), weightDequantizeOp, rewriter, _log.nest());

    if (getAlignmentFromConcatSlicePattern(origOp) > maxTileIC) {
        for (auto [tile, convOp] : llvm::zip_equal(tiles.value(), convOps)) {
            if (tile.shape[Dims4D::Act::C] == VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
                convOp->setAttr(preserveConcatAlignedICSplitAttr, rewriter.getUnitAttr());
            }
        }
    }

    rewriter.replaceOp(origOp, result);

    return mlir::success();
}

//
//  SplitMemPermuteAndDynamicDequantizeOp
//

class SplitMemPermuteAndDynamicDequantizeOp final : public mlir::OpRewritePattern<VPU::DynamicDequantizeOp> {
public:
    SplitMemPermuteAndDynamicDequantizeOp(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::DynamicDequantizeOp>(ctx), _log(log) {
        this->setDebugName("SplitMemPermuteAndDynamicDequantizeOp");
    }
    mlir::LogicalResult matchAndRewrite(VPU::DynamicDequantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SplitMemPermuteAndDynamicDequantizeOp::matchAndRewrite(VPU::DynamicDequantizeOp origOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    VPUX_THROW_UNLESS(origOp.getOutput().hasOneUse(), "DynamicDequantize output must have one use at '{0}'",
                      origOp->getLoc());

    auto memPermuteOp = origOp.getInput().getDefiningOp<VPU::MemPermuteOp>();
    if (memPermuteOp != nullptr && !memPermuteOp->hasOneUse()) {
        // Skip splitting MemPermuteOp if it has multiple uses
        memPermuteOp = nullptr;
    }

    SmallVector<mlir::Operation*> chain;
    auto user = *origOp.getOutput().getUsers().begin();
    while (mlir::isa_and_present<VPU::ViewLikeOpInterface>(user) && user->hasOneUse()) {
        chain.push_back(user);
        user = *user->getUsers().begin();
    }

    chain.push_back(user);

    auto sliceProducer = chain.back();
    SmallVector<VPU::SliceOp> sliceOps;
    for (auto sliceProducerUser : sliceProducer->getUsers()) {
        auto sliceOp = mlir::dyn_cast<VPU::SliceOp>(sliceProducerUser);
        VPUX_THROW_UNLESS(sliceOp != nullptr, "Expected Slice user for '{0}' at '{1}', got '{2}'",
                          sliceProducer->getName(), sliceProducer->getLoc(), sliceProducerUser->getName());
        sliceOps.push_back(sliceOp);
    }

    VPUX_THROW_UNLESS(!sliceOps.empty(), "Expected Slice users for '{0}' at '{1}'", sliceProducer->getName(),
                      sliceProducer->getLoc());

    const auto firstSliceInShape = getShape(sliceOps.front().getInput());
    const auto firstSliceOutShape = getShape(sliceOps.front().getResult());
    const auto maybeSliceAxis = IE::getSingleDiffAxis(firstSliceInShape, firstSliceOutShape);
    VPUX_THROW_UNLESS(maybeSliceAxis.has_value(), "Expected single slice axis for Slice at '{0}'",
                      sliceOps.front()->getLoc());
    const auto sliceAxis = maybeSliceAxis.value();

    struct SliceRewriteInfo final {
        VPU::SliceOp sliceOp;
        TileInfo dequantizeOutputTile;
        std::optional<TilingInfo> memPermuteInputTiling;
        SmallVector<TileInfo> viewOutputTiles;
        SmallVector<TilingInfo> viewInputTilings;
    };

    SmallVector<SliceRewriteInfo, 1> rewriteInfos;
    for (auto sliceOp : sliceOps) {
        const auto sliceInShape = getShape(sliceOp.getInput());
        const auto sliceOutShape = getShape(sliceOp.getResult());
        const auto curSliceAxis = IE::getSingleDiffAxis(sliceInShape, sliceOutShape);
        VPUX_THROW_UNLESS(curSliceAxis.has_value() && curSliceAxis.value() == sliceAxis,
                          "Expected Slice at '{0}' to use the same single axis '{1}'", sliceOp->getLoc(), sliceAxis);

        auto sliceOffsets = Shape(parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets()));
        auto sliceAxisInfo = Shape(sliceOutShape.size(), 1);
        // Use a placeholder tiling factor to mark the sliced dimension as tiled.
        sliceAxisInfo[sliceAxis] = 2;
        TileInfo curTile(sliceOutShape, sliceOffsets, sliceAxisInfo, true);

        SmallVector<TileInfo> viewOutputTiles;
        SmallVector<TilingInfo> viewInputTilings;
        viewOutputTiles.reserve(chain.size());
        viewInputTilings.reserve(chain.size());

        for (auto chainIdx : irange(chain.size()) | reversed) {
            auto viewLikeOp = mlir::cast<VPU::TilingViewLikeOpInterface>(chain[chainIdx]);
            viewOutputTiles.insert(viewOutputTiles.begin(), curTile);
            auto inputTiling = viewLikeOp.backInferTileInfo(curTile, _log.nest());
            VPUX_THROW_UNLESS(inputTiling.tiles.size() == 1,
                              "Expected one input tile for view-like op '{0}' at '{1}', got '{2}'",
                              viewLikeOp->getName(), viewLikeOp->getLoc(), inputTiling.tiles.size());
            viewInputTilings.insert(viewInputTilings.begin(), inputTiling);
            curTile = inputTiling.tiles.front();
        }

        std::optional<TilingInfo> memPermuteInputTiling;
        if (memPermuteOp != nullptr) {
            auto inputTiling = memPermuteOp.backInferTileInfo(curTile, _log.nest());
            VPUX_THROW_UNLESS(inputTiling.tiles.size() == 1,
                              "Expected one input tile for MemPermute at '{0}', got '{1}'", memPermuteOp->getLoc(),
                              inputTiling.tiles.size());
            memPermuteInputTiling = std::move(inputTiling);
        }

        rewriteInfos.push_back({sliceOp, std::move(curTile), std::move(memPermuteInputTiling),
                                std::move(viewOutputTiles), std::move(viewInputTilings)});
    }

    const auto createSliceForTile = [&](mlir::Value value, const TileInfo& tile, mlir::Location loc) -> mlir::Value {
        if (value == nullptr) {
            return nullptr;
        }

        const auto tileAxis = tile.axis.raw();
        const auto tiledAxisIt = llvm::find_if(tileAxis, [](int64_t axisTiles) {
            return axisTiles > 1;
        });
        if (tiledAxisIt == tileAxis.end()) {
            return value;
        }

        auto valueShape = getShape(value);
        const auto tiledAxis = std::distance(tileAxis.begin(), tiledAxisIt);
        if (valueShape[Dim(tiledAxis)] == 1) {
            return value;
        }

        auto offsets = SmallVector<int64_t>(valueShape.size(), 0);
        auto sizes = SmallVector<int64_t>(valueShape.raw().begin(), valueShape.raw().end());
        offsets[tiledAxis] = tile.offsets[Dim(tiledAxis)];
        sizes[tiledAxis] = tile.shape[Dim(tiledAxis)];

        return rewriter.create<VPU::SliceOp>(loc, value, offsets, sizes);
    };

    for (const auto& rewriteInfoIt : rewriteInfos | indexed) {
        const auto rewriteInfo = rewriteInfoIt.value();
        const auto index = rewriteInfoIt.index();

        auto sliceOp = rewriteInfo.sliceOp;
        rewriter.setInsertionPoint(sliceOp);

        mlir::Value inputSlice;
        if (memPermuteOp != nullptr) {
            auto memPermuteInputSlice =
                    createSliceForTile(memPermuteOp.getInput(), rewriteInfo.memPermuteInputTiling->tiles.front(),
                                       appendLoc(sliceOp.getLoc(), "_mempermute_{0}", index));

            mlir::IRMapping memPermuteMapper;
            memPermuteMapper.map(memPermuteOp.getInput(), memPermuteInputSlice);
            auto* newMemPermuteOp = rewriter.clone(*memPermuteOp, memPermuteMapper);
            newMemPermuteOp->setLoc(appendLoc(memPermuteOp->getLoc(), "_c_tile_{0}", index));

            auto newMemPermute = mlir::cast<VPU::TilingBuilderOpInterface>(newMemPermuteOp);
            newMemPermute.adjustAttrs(*rewriteInfo.memPermuteInputTiling, rewriteInfo.dequantizeOutputTile);
            vpux::inferReturnTypes(newMemPermuteOp, vpux::InferShapedTypeMode::ALL);

            inputSlice = newMemPermuteOp->getResult(0);
        } else {
            inputSlice = createSliceForTile(origOp.getInput(), rewriteInfo.dequantizeOutputTile,
                                            appendLoc(sliceOp.getLoc(), "_dequantize_in_{0}", index));
        }

        auto scaleSlice = createSliceForTile(origOp.getScale(), rewriteInfo.dequantizeOutputTile,
                                             appendLoc(sliceOp.getLoc(), "_dequantize_scale_{0}", index));
        auto zeroPointSlice = createSliceForTile(origOp.getZp(), rewriteInfo.dequantizeOutputTile,
                                                 appendLoc(sliceOp.getLoc(), "_dequantize_zp_{0}", index));

        auto newDequantizeOp =
                rewriter.create<VPU::DynamicDequantizeOp>(appendLoc(origOp.getLoc(), "_c_tile_{0}", index), inputSlice,
                                                          scaleSlice, zeroPointSlice, origOp.getDstElemTypeAttr());
        if (origOp.getMultiClusterStrategyAttr() != nullptr) {
            newDequantizeOp.setMultiClusterStrategyAttr(origOp.getMultiClusterStrategyAttr());
        }

        mlir::IRMapping mapper;
        mapper.map(origOp.getOutput(), newDequantizeOp.getOutput());
        mlir::Value curValue = newDequantizeOp.getOutput();
        for (auto chainIdx : irange(chain.size())) {
            mapper.map(chain[chainIdx]->getOperand(0), curValue);
            auto* newViewLikeOp = rewriter.clone(*chain[chainIdx], mapper);
            newViewLikeOp->setLoc(appendLoc(chain[chainIdx]->getLoc(), "_c_tile_{0}", index));

            auto newViewLike = mlir::cast<VPU::TilingViewLikeOpInterface>(newViewLikeOp);
            newViewLike.adjustAttrs(rewriteInfo.viewInputTilings[chainIdx], rewriteInfo.viewOutputTiles[chainIdx],
                                    getShape(chain[chainIdx]->getResult(0)));
            vpux::inferReturnTypes(newViewLikeOp, vpux::InferShapedTypeMode::ALL);

            curValue = newViewLikeOp->getResult(0);
            mapper.map(chain[chainIdx]->getResult(0), curValue);
        }

        rewriter.replaceOp(sliceOp, curValue);
    }

    for (auto oldViewLikeOp : chain | reversed) {
        rewriter.eraseOp(oldViewLikeOp);
    }

    rewriter.eraseOp(origOp);

    if (memPermuteOp != nullptr) {
        rewriter.eraseOp(memPermuteOp);
    }

    return mlir::success();
}

//
// EnsureNCEOpsSizeRequirementsPass
//

class EnsureNCEOpsSizeRequirementsPass final :
        public VPU::impl::EnsureNCEOpsSizeRequirementsBase<EnsureNCEOpsSizeRequirementsPass> {
public:
    explicit EnsureNCEOpsSizeRequirementsPass(bool enableOutputEnsurance,
                                              bool enableDequantWeightEnsuranceBeforeStrategy, SkipOCMode skipConvOC,
                                              SkipOCMode skipEltwiseOC, bool enableSplitChannelForDynamicDequantize,
                                              Logger log) {
        this->enableOutputEnsurance = enableOutputEnsurance;
        this->enableDequantWeightEnsuranceBeforeStrategy = enableDequantWeightEnsuranceBeforeStrategy;
        this->skipConvOC = skipConvOC;
        this->skipEltwiseOC = skipEltwiseOC;
        this->enableSplitChannelForDynamicDequantize = enableSplitChannelForDynamicDequantize;
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void EnsureNCEOpsSizeRequirementsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto moduleOp = func->getParentOfType<mlir::ModuleOp>();

    mlir::ConversionTarget target(ctx);
    mlir::RewritePatternSet patterns(&ctx);
    target.addLegalOp<VPU::SliceOp, VPU::ConcatOp>();

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        // TODO: #-196283 There is no pattern rewriter for the VPU.NCEMatMulOp,
        // it is better to catch the illegal operation and abort compilation process as soon as possible
        if (!mlir::isa<VPU::NCEConvolutionOp, VPU::NCEMatMulOp>(op)) {
            return true;
        }

        const auto inputShape = getShape(op->getOperand(0));
        if (mlir::isa<VPU::NCEMatMulOp>(op)) {
            return inputShape[DimsGroups5D::Act::C] <= VPU::NCEInvariant::VPU_DIMENSION_LIMIT;
        }

        // For NCEConvolutionOp use the dynamic pipelining-aware IC threshold.
        auto convOp = mlir::cast<VPU::NCEConvolutionOp>(op);
        const int64_t maxTileIC = computeMaxPipelinableIC(convOp);
        if (shouldSkipICSplitDueToConcatAlignment(convOp, maxTileIC)) {
            return true;
        }
        return inputShape[Dims4D::Act::C] <= maxTileIC;
    });

    patterns.add<EnsureConvICRequirements>(&ctx, enableSplitChannelForDynamicDequantize, _log);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }

    func.walk([](VPU::NCEConvolutionOp convOp) {
        convOp->removeAttr(preserveConcatAlignedICSplitAttr);
    });

    // If output shape ensurance is disabled, skip the rest of the pass
    // OC will be split at multi-cluster and tiling pass if needed
    if (!enableOutputEnsurance) {
        return;
    }

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (!mlir::isa<VPU::NCEOpInterface>(op)) {
            return true;
        }

        // Fused reduce outputs produce multiple results; tiling is not yet supported for such ops.
        // Always report as legal so no tiling is attempted.
        // TODO: E#209747 implement reduce output support after tiling, and remove this check
        if (op->getNumResults() > 1) {
            return true;
        }

        if (mlir::isa<VPU::TilingInfoOpInterface>(op)) {
            const auto inputShape = getShape(op->getOperand(0));
            const auto outputShape = getShape(op->getResult(0));
            const auto numClusters = config::getTileExecutor(moduleOp).getCount();

            // Construct dim-specific thresholds for input and output shapes
            // In our test, extending the threshold on Dim C can improve performance by reducing workloads for SOK NCE
            // operations when the number of clusters is greater than 2
            SmallVector<int64_t> inputDimThresholds(inputShape.size(), VPU::NCEInvariant::VPU_DIMENSION_LIMIT);
            SmallVector<int64_t> outputDimThresholds(outputShape.size(), VPU::NCEInvariant::VPU_DIMENSION_LIMIT);
            if (hasSplitOverKernelStrategy(op) && numClusters > 2) {
                inputDimThresholds[(Dims4D::Act::C).ind()] = VPU::NCEInvariant::VPU_DIMENSION_LIMIT * numClusters;
                outputDimThresholds[(Dims4D::Act::C).ind()] = VPU::NCEInvariant::VPU_DIMENSION_LIMIT * numClusters;
            }

            auto inSizeWrongDims = getDimsOverKHWLimit(inputShape, inputDimThresholds);
            if (!inSizeWrongDims.empty()) {
                _log.nest(2).debug("Input size has dims greater than HW requirements: {0}", inSizeWrongDims);
            }
            auto outSizeWrongDims = getDimsOverKHWLimit(outputShape, outputDimThresholds);
            if (!outSizeWrongDims.empty()) {
                _log.nest(2).debug("Output size has dims greater than HW requirements: {0}", outSizeWrongDims);
            }
            // Skip slicing conv with dequant weight input before strategy is assigned : this allows for more vertical
            // fusion for large convs
            if (enableDequantWeightEnsuranceBeforeStrategy) {
                if (auto convOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(op)) {
                    auto weightDequantizeOp = convOp.getFilter().getDefiningOp<VPU::DequantizeOp>();
                    if (weightDequantizeOp != nullptr) {
                        _log.nest(2).debug("Allow op {0} with dequant weights to skip dimension check before strategy "
                                           "assignment",
                                           op->getLoc());
                        return true;
                    }
                }
            }

            // Skip slicing C for per-channel based NCE ops, which will be handled later in tiling pass
            // This will benefit vertical fusion
            const auto eraseChannel = [&](SmallVector<Dim>& wrongDims) {
                wrongDims.erase(std::remove(wrongDims.begin(), wrongDims.end(), Dims4D::Act::C), wrongDims.end());
            };
            if (mlir::isa<VPU::NCEDepthConvolutionOp, VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp>(op)) {
                _log.nest(2).debug("Skip checking C dimension for per-channel based NCE op {0} at {1}", op->getName(),
                                   op->getLoc());
                eraseChannel(inSizeWrongDims);
                eraseChannel(outSizeWrongDims);
            }

            // For NCEConvolutionOp and NCEEltwiseOp, conditionally skip slicing OC based on mode:
            //   SKIP_NONE: always enforce OC limit.
            //   SKIP_LARGE_SPATIAL: skip OC check when H or W > 4.
            //   SKIP_ALL: always skip OC check.
            // TODO: Regressions (E#209583, E#209685, E#210083) block unconditional OC skipping. Remove the
            // spatial-gated workaround after those issues are fixed and switch the default mode for
            // VPU::NCEConvolutionOp and VPU::NCEEltwiseOp to SKIP_ALL.
            const auto applySkipOCMode = [&](SkipOCMode mode, bool isEltwise) {
                if (mode == SkipOCMode::SKIP_NONE) {
                    return;
                }
                const bool ocIsInWrongDims = llvm::is_contained(outSizeWrongDims, Dims4D::Act::C);
                if (!ocIsInWrongDims) {
                    return;
                }
                constexpr int64_t kSpatialLimit = 4;
                const bool largeSpatial =
                        outputShape[Dims4D::Act::H] > kSpatialLimit || outputShape[Dims4D::Act::W] > kSpatialLimit;
                const bool doSkip =
                        (mode == SkipOCMode::SKIP_ALL) || (mode == SkipOCMode::SKIP_LARGE_SPATIAL && largeSpatial);
                if (doSkip) {
                    _log.nest(2).debug("Skip checking OC dimension for {0} at {1} (mode={2})", op->getName(),
                                       op->getLoc(), mode);
                    eraseChannel(outSizeWrongDims);
                    if (isEltwise) {
                        eraseChannel(inSizeWrongDims);
                    }
                }
            };
            if (mlir::isa<VPU::NCEConvolutionOp>(op)) {
                applySkipOCMode(skipConvOC, /*isEltwise=*/false);
            } else if (mlir::isa<VPU::NCEEltwiseOp>(op)) {
                applySkipOCMode(skipEltwiseOC, /*isEltwise=*/true);
            }

            return inSizeWrongDims.empty() && outSizeWrongDims.empty();
        }

        return true;
    });

    mlir::RewritePatternSet nceSizeRequirementPatterns(&ctx);
    nceSizeRequirementPatterns.add<EnsureNCEOpSizeRequirements>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(nceSizeRequirementPatterns)))) {
        signalPassFailure();
    }

    // `SplitMemPermuteAndDynamicDequantizeOp` is only applicable for the platforms that support
    // `DynamicDequantConvVFPattern`
    if (!enableSplitChannelForDynamicDequantize) {
        return;
    }

    target.addDynamicallyLegalOp<VPU::DynamicDequantizeOp>([&](VPU::DynamicDequantizeOp dynamicDQ) {
        if (!dynamicDQ.getOutput().hasOneUse()) {
            return true;
        }

        SmallVector<mlir::Operation*> chain;
        auto curOp = *dynamicDQ.getOutput().getUsers().begin();
        while (mlir::isa_and_present<VPU::ViewLikeOpInterface>(curOp) && curOp->hasOneUse()) {
            chain.push_back(curOp);
            curOp = *curOp->getUsers().begin();
        }

        if (curOp->use_empty() || !mlir::isa_and_present<VPU::ViewLikeOpInterface>(curOp)) {
            return true;
        }
        chain.push_back(curOp);

        auto firstSliceOp = mlir::dyn_cast_if_present<VPU::SliceOp>(*curOp->getUsers().begin());
        if (firstSliceOp == nullptr) {
            return true;
        }

        const auto firstSliceInShape = getShape(firstSliceOp.getInput());
        const auto firstSliceOutShape = getShape(firstSliceOp.getResult());
        const auto firstSliceAxis = IE::getSingleDiffAxis(firstSliceInShape, firstSliceOutShape);
        if (!firstSliceAxis.has_value()) {
            return true;
        }

        // AffineReshape/ShapeCast can back-infer a slice only as multiple rectangular input tiles. e.g,
        // AffineReshape [1, 3584, 148, 128] -> [3584, 18944, 1, 1] maps output dim1 to input dims 2 and 3;
        // a [3584, 6320, 1, 1] slice spans 49 full rows and 48 elements of the next row will fail the back-inference.
        const auto isSupportedByViewChain = [&](VPU::SliceOp sliceOp) {
            const auto sliceOutShape = getShape(sliceOp.getResult());
            auto sliceOffsets = Shape(parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets()));
            auto sliceAxisInfo = Shape(sliceOutShape.size(), 1);
            sliceAxisInfo[firstSliceAxis.value()] = 2;
            TileInfo curTile(sliceOutShape, sliceOffsets, sliceAxisInfo, true);

            for (auto chainIdx : irange(chain.size()) | reversed) {
                auto viewLikeOp = mlir::dyn_cast<VPU::TilingViewLikeOpInterface>(chain[chainIdx]);
                if (viewLikeOp == nullptr || !viewLikeOp.isSupportedOutTile(curTile)) {
                    return false;
                }

                auto inputTiling = viewLikeOp.backInferTileInfo(curTile, _log.nest());
                if (inputTiling.tiles.size() != 1) {
                    return false;
                }
                curTile = inputTiling.tiles.front();
            }

            return true;
        };

        return !llvm::all_of(curOp->getUsers(), [&](auto sliceProducerUser) {
            auto sliceOp = mlir::dyn_cast<VPU::SliceOp>(sliceProducerUser);
            if (sliceOp == nullptr) {
                return false;
            }

            const auto sliceInShape = getShape(sliceOp.getInput());
            const auto sliceOutShape = getShape(sliceOp.getResult());
            const auto sliceAxis = IE::getSingleDiffAxis(sliceInShape, sliceOutShape);
            return sliceAxis.has_value() && sliceAxis.value() == firstSliceAxis.value() &&
                   isSupportedByViewChain(sliceOp);
        });
    });

    mlir::RewritePatternSet splitDynamicDequantizePatterns(&ctx);
    // Convert pattern
    //                                                           -> Slice
    // MemPermute (optional) -> DynamicDequantize -> ViewLikeOps -> Slice
    //                                                           -> Slice
    // To
    //
    // Slice -> MemPermute (optional) -> DynamicDequantize -> ViewLikeOps
    // Slice -> MemPermute (optional) -> DynamicDequantize -> ViewLikeOps
    // Slice -> MemPermute (optional) -> DynamicDequantize -> ViewLikeOps
    splitDynamicDequantizePatterns.add<SplitMemPermuteAndDynamicDequantizeOp>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(splitDynamicDequantizePatterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createEnsureNCEOpsSizeRequirementsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createEnsureNCEOpsSizeRequirementsPass(
        bool enableOutputEnsurance, bool enableDequantWeightEnsuranceBeforeStrategy, SkipOCMode skipConvOC,
        SkipOCMode skipEltwiseOC, bool enableSplitChannelForDynamicDequantize, Logger log) {
    return std::make_unique<EnsureNCEOpsSizeRequirementsPass>(
            enableOutputEnsurance, enableDequantWeightEnsuranceBeforeStrategy, skipConvOC, skipEltwiseOC,
            enableSplitChannelForDynamicDequantize, log);
}
