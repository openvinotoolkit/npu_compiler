//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/multicluster_tiling_scf_algorithm.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/scf_tiling/scf_tiling.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/TileUsingInterface.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Support/LLVM.h>

using namespace vpux;
using namespace VPU;

namespace {
Shape getTilesFromStrategy(mlir::Operation* op, int64_t numClusters, VPU::MultiClusterStrategy strategy) {
    const auto outType = mlir::dyn_cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    Shape tilesOnDim(outType.getRank(), 1);
    switch (strategy) {
    case VPU::MultiClusterStrategy::Clustering:
        break;
    case VPU::MultiClusterStrategy::SplitOverKernel:
        tilesOnDim[Dims4D::Act::C] = numClusters;
        break;
    case VPU::MultiClusterStrategy::SplitOverHeight:
    case VPU::MultiClusterStrategy::SplitOverHeightOverlapped:
    case VPU::MultiClusterStrategy::HKSwitch:
        // Treat HKSwitch as SOH as first step
        // TODO: E#193460 Add broadcast capability
        tilesOnDim[Dims4D::Act::H] = numClusters;
        break;
    case VPU::MultiClusterStrategy::SplitOverWidth:
        tilesOnDim[Dims4D::Act::W] = numClusters;
        break;
    case VPU::MultiClusterStrategy::SplitOverBatch:
        tilesOnDim[Dims4D::Act::N] = numClusters;
        break;
    case VPU::MultiClusterStrategy::SplitOverGroup:
        tilesOnDim[Dims5D::Act::N] = numClusters;
        break;
    default:
        VPUX_THROW("Unsupported strategy for getTilesFromStrategy: {0}", strategy);
    }
    return tilesOnDim;
}

// E-162999 workaround: tileUsingSCF and manual forall construction drop the
// output order attribute. Restore original output types on the forall results,
// shared outputs, and ParallelInsertSlice destinations.
void fixForallOutputTypes(mlir::scf::ForallOp forallOp, mlir::Operation* origOp) {
    for (unsigned idx = 0; idx < origOp->getNumResults(); ++idx) {
        forallOp.getResult(idx).setType(origOp->getResult(idx).getType());
    }
    auto* terminator = forallOp.getBody()->getTerminator();
    if (auto inParallelOp = mlir::dyn_cast_or_null<mlir::scf::InParallelOp>(terminator)) {
        for (auto insertOp : inParallelOp.getOps<mlir::tensor::ParallelInsertSliceOp>()) {
            if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(insertOp.getDest())) {
                const auto argNum = blockArg.getArgNumber();
                const auto inductionVarCount = forallOp.getInductionVars().size();
                VPUX_THROW_UNLESS(argNum >= inductionVarCount,
                                  "ParallelInsertSlice destination block argument index {0} is less than "
                                  "induction variable count {1}; malformed scf.forall IR",
                                  argNum, inductionVarCount);
                auto argIndex = argNum - inductionVarCount;
                auto outputType = origOp->getResult(argIndex).getType();
                insertOp.getDestMutable().get().setType(outputType);
                forallOp.getOutputs()[argIndex].setType(outputType);
            }
        }
    }
}

struct BalancedTileSizes {
    int64_t largeTile;
    int64_t smallTile;
    int64_t numLarge;  // Number of iterations that use largeTile.
};

// Compute balanced tile sizes when a dimension cannot be divided uniformly
// across clusters. The remainder is spread evenly: numLarge iterations get
// largeTile (= smallTile + alignment) while the rest get smallTile.
// This minimizes the maximum tile size, which determines runtime.
// Example: dimSize=64, clusters=3, alignment=16 -> {32, 16, 1} for [32, 16, 16].
// Example: dimSize=27, clusters=6, alignment=1  -> {5, 4, 3}  for [5, 5, 5, 4, 4, 4].
// Example: dimSize=96, clusters=4, alignment=16 -> {32, 16, 2} for [32, 32, 16, 16].
std::optional<BalancedTileSizes> getBalancedTileSizes(int64_t dimSize, int64_t numClusters, int64_t alignment) {
    if (alignment <= 0 || numClusters <= 1) {
        return std::nullopt;
    }
    const auto smallTile = (dimSize / numClusters / alignment) * alignment;
    if (smallTile <= 0) {
        return std::nullopt;
    }
    const auto remainder = dimSize - smallTile * numClusters;
    if (remainder == 0) {
        return std::nullopt;  // Uniform division; balanced split is not needed.
    }
    if (remainder < 0 || remainder % alignment != 0) {
        return std::nullopt;
    }
    const auto numLarge = remainder / alignment;
    const auto largeTile = smallTile + alignment;
    return BalancedTileSizes{largeTile, smallTile, numLarge};
}

// Unified balanced multiclustering implementation. Distributes an output dimension
// unevenly across numClusters iterations using normalized loop bounds (0 to numClusters
// step 1). The iv is the cluster index; per-iteration offset and size are computed from
// pre-built values for smallTile, numLarge, and delta (= largeTile - smallTile = alignment).
// This helper can consume MLIR values, but in the current balanced-tiling flow these
// values must already be constant or foldable when the validity checks are performed;
// arbitrary runtime-computed MC dimension sizes are not supported here. After unrolling,
// the per-iteration arithmetic folds to constants for the supported cases.
mlir::LogicalResult applyBalancedMulticlusterTilingImpl(mlir::Operation* operation, mlir::RewriterBase& builder,
                                                        int64_t numClusters, Dim mcDim, mlir::Value smallTileVal,
                                                        mlir::Value numLargeVal, mlir::Value deltaVal, Logger /*log*/) {
    auto tilingOp = mlir::cast<mlir::TilingInterface>(operation);
    auto loc = operation->getLoc();
    const auto numResults = operation->getNumResults();
    auto primaryOutputType = mlir::cast<mlir::RankedTensorType>(operation->getResult(0).getType());
    const auto rank = primaryOutputType.getRank();

    // Create tensor.empty for each result, collecting dynamic dims where needed.
    mlir::ReifiedRankedShapedTypeDims reifiedShapes;
    const bool hasReified = mlir::succeeded(mlir::reifyResultShapes(builder, operation, reifiedShapes));

    SmallVector<mlir::Value> outputEmpties;
    outputEmpties.reserve(numResults);
    for (unsigned idx = 0; idx < numResults; ++idx) {
        auto resultType = mlir::cast<mlir::RankedTensorType>(operation->getResult(idx).getType());
        SmallVector<mlir::Value> dynamicDims;
        for (int64_t d = 0; d < resultType.getRank(); ++d) {
            if (resultType.isDynamicDim(d)) {
                if (hasReified) {
                    dynamicDims.push_back(mlir::getValueOrCreateConstantIndexOp(builder, loc, reifiedShapes[idx][d]));
                } else {
                    dynamicDims.push_back(
                            builder.create<mlir::tensor::DimOp>(loc, operation->getResult(idx), d).getResult());
                }
            }
        }
        auto emptyOp = builder.create<mlir::tensor::EmptyOp>(loc, resultType, dynamicDims);
        outputEmpties.push_back(emptyOp.getResult());
    }

    // Normalized forall: (0) to (numClusters) step (1). The iv is the cluster index.
    SmallVector<mlir::OpFoldResult> lbs = {builder.getIndexAttr(0)};
    SmallVector<mlir::OpFoldResult> ubs = {builder.getIndexAttr(numClusters)};
    SmallVector<mlir::OpFoldResult> steps = {builder.getIndexAttr(1)};

    auto forallOp =
            builder.create<mlir::scf::ForallOp>(loc, lbs, ubs, steps, mlir::ValueRange{outputEmpties}, std::nullopt);

    // Body: iv is the cluster index (0, 1, ..., numClusters-1)
    //   extraTiles = min(iv, numLarge)
    //   realOffset = iv * smallTile + extraTiles * delta
    //   isLarge    = iv < numLarge
    //   realSize   = isLarge ? (smallTile + delta) : smallTile
    builder.setInsertionPointToStart(forallOp.getBody());
    auto iv = forallOp.getInductionVar(0);

    auto extraTiles = builder.create<mlir::arith::MinUIOp>(loc, iv, numLargeVal);
    auto baseOffset = builder.create<mlir::arith::MulIOp>(loc, iv, smallTileVal);
    auto extraOffset = builder.create<mlir::arith::MulIOp>(loc, extraTiles, deltaVal);
    auto realOffset = builder.create<mlir::arith::AddIOp>(loc, baseOffset, extraOffset);

    auto isLarge = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, iv, numLargeVal);
    auto largeTileVal = builder.create<mlir::arith::AddIOp>(loc, smallTileVal, deltaVal);
    auto realSize = builder.create<mlir::arith::SelectOp>(loc, isLarge, largeTileVal, smallTileVal);

    // Build full-rank offsets and sizes for TilingInterface::getTiledImplementation.
    SmallVector<mlir::OpFoldResult> tileOffsets(rank, builder.getIndexAttr(0));
    SmallVector<mlir::OpFoldResult> tileSizes;
    for (int64_t i = 0; i < rank; ++i) {
        if (i == mcDim.ind()) {
            tileOffsets[i] = realOffset.getResult();
            tileSizes.push_back(realSize.getResult());
        } else {
            if (primaryOutputType.isDynamicDim(i)) {
                if (hasReified) {
                    tileSizes.push_back(reifiedShapes[0][i]);
                } else {
                    tileSizes.push_back(
                            builder.create<mlir::tensor::DimOp>(loc, operation->getResult(0), i).getResult());
                }
            } else {
                tileSizes.push_back(builder.getIndexAttr(primaryOutputType.getDimSize(i)));
            }
        }
    }

    auto erasePartialIR = [&]() {
        builder.eraseOp(forallOp);
        for (auto it = outputEmpties.rbegin(); it != outputEmpties.rend(); ++it) {
            builder.eraseOp(it->getDefiningOp());
        }
    };

    auto tiledResult = tilingOp.getTiledImplementation(builder, tileOffsets, tileSizes);
    if (mlir::failed(tiledResult) || tiledResult->tiledValues.empty()) {
        erasePartialIR();
        return mlir::failure();
    }

    auto inParallelOp = forallOp.getTerminator();
    builder.setInsertionPointToStart(inParallelOp.getBody());
    auto regionOutArgs = forallOp.getRegionOutArgs();

    for (unsigned idx = 0; idx < numResults; ++idx) {
        SmallVector<mlir::OpFoldResult> resultOffsets, resultSizes;
        if (mlir::failed(
                    tilingOp.getResultTilePosition(builder, idx, tileOffsets, tileSizes, resultOffsets, resultSizes))) {
            erasePartialIR();
            return mlir::failure();
        }
        auto resultRank = mlir::cast<mlir::RankedTensorType>(operation->getResult(idx).getType()).getRank();
        SmallVector<mlir::OpFoldResult> strides(resultRank, builder.getIndexAttr(1));
        builder.create<mlir::tensor::ParallelInsertSliceOp>(loc, tiledResult->tiledValues[idx], regionOutArgs[idx],
                                                            resultOffsets, resultSizes, strides);
    }

    fixForallOutputTypes(forallOp, operation);

    builder.replaceOp(operation, forallOp.getResults());
    return mlir::success();
}

struct BalancedTileValues {
    mlir::Value smallTile;
    mlir::Value numLarge;
    mlir::Value delta;
};

// Compute balanced tile parameters from the MC dimension size (static or dynamic).
// For static dimensions, validates feasibility via getBalancedTileSizes and emits constants.
// For dynamic dimensions, emits affine/arith SSA computation; validates that the remainder
// folds to constants satisfying alignment constraints.
// Returns nullopt if balanced distribution is not applicable.
std::optional<BalancedTileValues> computeBalancedTileValues(mlir::RewriterBase& builder, mlir::Location loc,
                                                            mlir::OpFoldResult dimSize, int64_t numClusters,
                                                            int64_t alignment) {
    // Static path: extract integer, validate, emit constants.
    if (auto attr = mlir::dyn_cast_if_present<mlir::Attribute>(dimSize)) {
        auto dimVal = mlir::cast<mlir::IntegerAttr>(attr).getInt();
        auto balanced = getBalancedTileSizes(dimVal, numClusters, alignment);
        if (!balanced) {
            return std::nullopt;
        }
        const int64_t delta = balanced->largeTile - balanced->smallTile;
        return BalancedTileValues{builder.create<mlir::arith::ConstantIndexOp>(loc, balanced->smallTile),
                                  builder.create<mlir::arith::ConstantIndexOp>(loc, balanced->numLarge),
                                  builder.create<mlir::arith::ConstantIndexOp>(loc, delta)};
    }

    // Dynamic path: emit runtime computation.
    //   smallTile = (dimSize / numClusters / alignment) * alignment
    //   remainder = dimSize - smallTile * numClusters
    //   numLarge  = remainder / alignment
    //   delta     = alignment
    auto dimSizeVal = mlir::cast<mlir::Value>(dimSize);

    mlir::AffineExpr d0;
    bindDims(builder.getContext(), d0);
    auto smallTileMap = mlir::AffineMap::get(1, 0, {d0.floorDiv(numClusters).floorDiv(alignment) * alignment},
                                             builder.getContext());
    auto smallTileOFR = mlir::affine::makeComposedFoldedAffineApply(builder, appendLoc(loc, "smallTile"), smallTileMap,
                                                                    {dimSizeVal});
    auto smallTileVal = mlir::getValueOrCreateConstantIndexOp(builder, loc, smallTileOFR);

    auto eraseIfUnused = [](mlir::Value value) {
        if (auto* defOp = value.getDefiningOp(); defOp != nullptr && defOp->use_empty()) {
            defOp->erase();
        }
    };

    auto numClustersConst = builder.create<mlir::arith::ConstantIndexOp>(loc, numClusters);
    auto totalSmall = builder.createOrFold<mlir::arith::MulIOp>(loc, smallTileVal, numClustersConst);
    auto remainder = builder.createOrFold<mlir::arith::SubIOp>(loc, dimSizeVal, totalSmall);
    auto deltaVal = builder.create<mlir::arith::ConstantIndexOp>(loc, alignment);
    auto zeroVal = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);

    // Validate: all checks must fold to true constants; otherwise the balanced split is not applicable.
    // Equivalent to the static path guards:
    //   smallTile > 0   — rejects cases where dimSize < numClusters * alignment (zero-sized tiles)
    //   remainder != 0  — rejects uniform splits (static path returns nullopt when remainder == 0)
    //   remainder >= 0  — rejects underflow
    //   remainder % alignment == 0 — rejects misaligned remainders
    auto smallTilePositive =
            builder.createOrFold<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, smallTileVal, zeroVal);
    auto remainderNonZero =
            builder.createOrFold<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, remainder, zeroVal);
    auto remainderNonNegative =
            builder.createOrFold<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, remainder, zeroVal);
    auto remainderModAlignment = builder.createOrFold<mlir::arith::RemUIOp>(loc, remainder, deltaVal);
    auto remainderAligned = builder.createOrFold<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq,
                                                                      remainderModAlignment, zeroVal);

    auto getConstantIntValue = [](mlir::Value value) -> std::optional<int64_t> {
        if (auto constIndexOp = value.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
            return constIndexOp.value();
        }
        if (auto constIntOp = value.getDefiningOp<mlir::arith::ConstantIntOp>()) {
            return constIntOp.value();
        }
        return std::nullopt;
    };

    auto smallTilePosConst = getConstantIntValue(smallTilePositive);
    auto nonZeroConst = getConstantIntValue(remainderNonZero);
    auto nonNegConst = getConstantIntValue(remainderNonNegative);
    auto alignedConst = getConstantIntValue(remainderAligned);
    if (!smallTilePosConst || !nonZeroConst || !nonNegConst || !alignedConst || *smallTilePosConst == 0 ||
        *nonZeroConst == 0 || *nonNegConst == 0 || *alignedConst == 0) {
        eraseIfUnused(remainderAligned);
        eraseIfUnused(remainderModAlignment);
        eraseIfUnused(remainderNonNegative);
        eraseIfUnused(remainderNonZero);
        eraseIfUnused(smallTilePositive);
        eraseIfUnused(zeroVal);
        eraseIfUnused(deltaVal);
        eraseIfUnused(remainder);
        eraseIfUnused(totalSmall);
        eraseIfUnused(numClustersConst);
        eraseIfUnused(smallTileVal);
        return std::nullopt;
    }

    auto numLargeVal = builder.createOrFold<mlir::arith::DivUIOp>(loc, remainder, deltaVal);
    return BalancedTileValues{smallTileVal, numLargeVal, deltaVal};
}

};  // namespace

mlir::LogicalResult MulticlusterTilingSCFAlgorithm::applyTiling(mlir::Operation* operation, mlir::RewriterBase& builder,
                                                                Logger log) {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation);
    if (clusteredOp == nullptr) {
        log.trace("Op is not cluster-able.");
        return mlir::failure();
    }

    if (!clusteredOp.getMultiClusterStrategy().has_value()) {
        log.trace("Op has no multiclustering strategy assigned.");
        return mlir::failure();
    }

    const auto mcStrategy = clusteredOp.getMultiClusterStrategy().value();
    // TODO: E#193453 num clusters may be different based on the size of the tile, if MC axis is the
    // same as tiling axis
    const auto outShape = getBoundedShape(clusteredOp->getResult(0));
    const auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outShape, mcStrategy);
    const auto strategy = getTilesFromStrategy(operation, numClusters, mcStrategy);

    // Clustering replicates data across all clusters without splitting dimension.
    // Generate scf.forall with numClusters iterations where each iteration runs the
    // full operation. This representation is needed for future vertical fusion support.
    // To avoid UB from overlapping parallel_insert_slice writes (all iterations writing
    // to [0,0,0,0]), expand dim 0 by numClusters so each iteration writes a disjoint
    // slice. After the forall, extract_slice recovers the original shape.
    if (mcStrategy == VPU::MultiClusterStrategy::Clustering) {
        log.trace("Generating scf.forall for Clustering strategy with {0} clusters.", numClusters);

        builder.setInsertionPoint(operation);

        auto loc = operation->getLoc();
        const auto numResults = operation->getNumResults();
        constexpr int64_t expandDimIdx = 0;

        // Compute original output sizes and create expanded tensor.empty for each result.
        // This is similar to tensor::getOrCreateDestinations, but we need to preserve
        // the encoding (e.g., layout order, bounds) when creating tensor.empty.
        // TODO: E-210745 Extend tensor::getOrCreateDestinations to Preserve Tensor Encoding.
        mlir::ReifiedRankedShapedTypeDims reifiedShapes;
        const bool hasReified = mlir::succeeded(mlir::reifyResultShapes(builder, operation, reifiedShapes));

        SmallVector<mlir::Value> outputEmpties;
        SmallVector<SmallVector<mlir::OpFoldResult>> allOrigSizes;
        outputEmpties.reserve(numResults);
        allOrigSizes.reserve(numResults);

        for (unsigned idx = 0; idx < numResults; ++idx) {
            auto resultType = mlir::cast<mlir::RankedTensorType>(operation->getResult(idx).getType());

            SmallVector<mlir::OpFoldResult> origSizes;
            if (resultType.hasStaticShape()) {
                for (auto dim : llvm::seq<int64_t>(0, resultType.getRank())) {
                    origSizes.push_back(builder.getIndexAttr(resultType.getDimSize(dim)));
                }
            } else if (hasReified) {
                origSizes = reifiedShapes[idx];
            } else {
                return operation->emitError("Clustering requires static shapes or reifyResultShapes support");
            }

            // Expand dim 0: origDim0 * numClusters.
            // Use origSizes[0] (the already-built OpFoldResult) so dynamic sizes are handled
            // correctly. If static, fold to a constant; otherwise emit arith.muli.
            SmallVector<mlir::OpFoldResult> expandedSizes = origSizes;
            if (auto attr = llvm::dyn_cast_if_present<mlir::Attribute>(origSizes[expandDimIdx])) {
                auto dim0 = mlir::dyn_cast<mlir::IntegerAttr>(attr).getInt();
                expandedSizes[expandDimIdx] = builder.getIndexAttr(dim0 * numClusters);
            } else {
                auto dim0Val = mlir::getValueOrCreateConstantIndexOp(builder, loc, origSizes[expandDimIdx]);
                auto numClustersVal = builder.create<mlir::arith::ConstantIndexOp>(loc, numClusters);
                expandedSizes[expandDimIdx] =
                        builder.create<mlir::arith::MulIOp>(loc, dim0Val, numClustersVal).getResult();
            }

            // Build expanded encoding: if the original has bounds, update dim 0 bound.
            auto expandedEncoding = resultType.getEncoding();
            if (auto tensorAttr = mlir::dyn_cast_if_present<vpux::TensorAttr>(expandedEncoding)) {
                auto origBounds = tensorAttr.getBounds();
                if (!origBounds.empty()) {
                    auto expandedBounds = Bounds(origBounds.raw());
                    expandedBounds[Dim(expandDimIdx)] = origBounds[Dim(expandDimIdx)] * numClusters;
                    expandedEncoding =
                            vpux::getTensorAttr(builder.getContext(), tensorAttr.getOrder(), tensorAttr.getMemSpace(),
                                                expandedBounds, tensorAttr.getDynamicDimsMask());
                }
            }

            auto emptyOp = builder.create<mlir::tensor::EmptyOp>(loc, expandedSizes, resultType.getElementType(),
                                                                 expandedEncoding);
            outputEmpties.push_back(emptyOp.getResult());
            allOrigSizes.push_back(std::move(origSizes));
        }

        builder.modifyOpInPlace(operation, [&] {
            operation->removeAttr(multiClusterStrategy);
        });

        // Create scf.forall with numClusters iterations (normalized loop: only upper bounds)
        SmallVector<mlir::OpFoldResult> upperBounds = {builder.getIndexAttr(numClusters)};

        auto forallOp = builder.create<mlir::scf::ForallOp>(loc, upperBounds,
                                                            /*outputs=*/mlir::ValueRange{outputEmpties},
                                                            /*mapping=*/std::nullopt);

        auto* terminator = forallOp.getBody()->getTerminator();
        builder.setInsertionPoint(terminator);

        // Compute per-cluster offset in dim 0: iv * origDim0.
        // Derive step from allOrigSizes (the pre-built OpFoldResults) so dynamic dim0 is correct.
        auto iv = forallOp.getBody()->getArgument(0);
        SmallVector<mlir::Value> dim0Offsets;
        dim0Offsets.reserve(numResults);
        for (unsigned idx = 0; idx < numResults; ++idx) {
            auto origDim0OFR = allOrigSizes[idx][expandDimIdx];
            // Fast path: static size 1 — offset is just the induction variable.
            if (auto attr = llvm::dyn_cast_if_present<mlir::Attribute>(origDim0OFR)) {
                if (mlir::dyn_cast<mlir::IntegerAttr>(attr).getInt() == 1) {
                    dim0Offsets.push_back(iv);
                    continue;
                }
            }
            auto step = mlir::getValueOrCreateConstantIndexOp(builder, loc, origDim0OFR);
            dim0Offsets.push_back(builder.create<mlir::arith::MulIOp>(loc, iv, step).getResult());
        }

        // Clone the original operation inside the forall
        auto* clonedOp = builder.clone(*operation);

        // Create parallel_insert_slice for each result — disjoint writes along dim 0
        auto inParallelOp = mlir::cast<mlir::scf::InParallelOp>(terminator);
        builder.setInsertionPointToStart(inParallelOp.getBody());

        auto regionOutArgs = forallOp.getRegionOutArgs();
        for (unsigned idx = 0; idx < numResults; ++idx) {
            auto resultType = mlir::cast<mlir::RankedTensorType>(operation->getResult(idx).getType());
            auto rank = resultType.getRank();
            SmallVector<mlir::OpFoldResult> offsets(rank, builder.getIndexAttr(0));
            offsets[expandDimIdx] = dim0Offsets[idx];
            SmallVector<mlir::OpFoldResult> strides(rank, builder.getIndexAttr(1));

            builder.create<mlir::tensor::ParallelInsertSliceOp>(loc, clonedOp->getResult(idx), regionOutArgs[idx],
                                                                offsets, allOrigSizes[idx], strides);
        }

        // Fix forall output types to include encoding from expanded shape.
        // For dynamic dim0 keep kDynamic; the runtime size comes from the tensor.empty operand.
        // Update bounds in encoding to match expanded dim 0.
        for (unsigned idx = 0; idx < numResults; ++idx) {
            auto origType = mlir::cast<mlir::RankedTensorType>(operation->getResult(idx).getType());
            auto expandedShape = llvm::to_vector(origType.getShape());
            if (expandedShape[expandDimIdx] != mlir::ShapedType::kDynamic) {
                expandedShape[expandDimIdx] *= numClusters;
            }
            auto expandedEncoding = origType.getEncoding();
            if (auto tensorAttr = mlir::dyn_cast_if_present<vpux::TensorAttr>(expandedEncoding)) {
                auto origBounds = tensorAttr.getBounds();
                if (!origBounds.empty()) {
                    auto expandedBounds = Bounds(origBounds.raw());
                    expandedBounds[Dim(expandDimIdx)] = origBounds[Dim(expandDimIdx)] * numClusters;
                    expandedEncoding =
                            vpux::getTensorAttr(builder.getContext(), tensorAttr.getOrder(), tensorAttr.getMemSpace(),
                                                expandedBounds, tensorAttr.getDynamicDimsMask());
                }
            }
            auto expandedType = mlir::RankedTensorType::get(expandedShape, origType.getElementType(), expandedEncoding);
            forallOp.getResult(idx).setType(expandedType);
            regionOutArgs[idx].setType(expandedType);
        }

        // Extract one copy from each expanded result to recover original shape
        builder.setInsertionPointAfter(forallOp);
        SmallVector<mlir::Value> replacements;
        for (unsigned idx = 0; idx < numResults; ++idx) {
            auto origType = mlir::cast<mlir::RankedTensorType>(operation->getResult(idx).getType());
            auto rank = origType.getRank();
            SmallVector<mlir::OpFoldResult> offsets(rank, builder.getIndexAttr(0));
            SmallVector<mlir::OpFoldResult> strides(rank, builder.getIndexAttr(1));

            auto extractOp = builder.create<mlir::tensor::ExtractSliceOp>(loc, origType, forallOp.getResult(idx),
                                                                          offsets, allOrigSizes[idx], strides);
            replacements.push_back(extractOp.getResult());
        }
        builder.replaceOp(operation, replacements);
        return mlir::success();
    }

    // Workaround to re-use current infrastructure from tiling.
    // TODO: E#192457 Implement proper multiclustering
    builder.modifyOpInPlace(operation, [&]() {
        operation->removeAttr(multiClusterStrategy);
        operation->setAttr(tilingStrategy, getIntArrayAttr(builder, strategy));
    });

    mlir::scf::SCFTilingOptions tilingOptions;

    const auto mcAxis = VPU::getDistributedTilingAxis(strategy.raw());
    const auto tileSizeComputationFnc = [&](mlir::OpBuilder&, mlir::Operation*) {
        const auto outShape = getShape(operation->getResult(0));
        if (outShape.isDynamic()) {
            return dynamicTileSizeComputation(builder, {operation}, nullptr, strategy,
                                              outShape[Dim(mcAxis)] != mlir::ShapedType::kDynamic);
        }

        std::unordered_map<Dim, std::pair<int64_t, int64_t>> emptyRemainders;

        return staticTileSizeComputation(builder, {operation}, nullptr, strategy, getShape(operation->getResult(0)),
                                         emptyRemainders);
    };

    tilingOptions.setTileSizeComputationFunction(tileSizeComputationFnc);
    tilingOptions.setLoopType(mlir::scf::SCFTilingOptions::LoopType::ForallOp);

    auto tilingResult = mlir::scf::tileUsingSCF(builder, mlir::cast<mlir::TilingInterface>(operation), tilingOptions);
    if (mlir::failed(tilingResult) || tilingResult->loops.empty()) {
        // Uniform tiling failed (e.g. alignment prevents even division).
        // Fall back to balanced distribution: the first `numLarge` iterations use `largeTile`
        // and the remaining iterations use `smallTile` (e.g. [32, 32, 16, 16], not a single
        // enlarged first tile followed by all-small tiles).
        // The fallback derives the iteration space from result(0). For multi-result ops,
        // all results must share the same rank, mc-dimension size, and all non-mc dimension
        // sizes, because applyBalancedMulticlusterTilingImpl() uses result(0)'s non-mc sizes
        const auto currentOutShape = getShape(operation->getResult(0));
        const auto allResultsCompatible = [&]() {
            const auto primaryRank = currentOutShape.size();
            for (unsigned idx = 1; idx < operation->getNumResults(); ++idx) {
                const auto shape = getShape(operation->getResult(idx));
                if (shape.size() != primaryRank) {
                    return false;
                }
                for (size_t dim = 0; dim < primaryRank; ++dim) {
                    if (shape[Dim(dim)] != currentOutShape[Dim(dim)]) {
                        return false;
                    }
                }
            }
            return true;
        }();
        if (!allResultsCompatible) {
            return operation->emitError("Tiling algorithm failed");
        }

        builder.setInsertionPoint(operation);
        auto loc = operation->getLoc();
        const auto boundedShape = getBoundedShape(operation->getResult(0));
        const auto alignment = vpux::getAlignment(operation, strategy, boundedShape);

        mlir::OpFoldResult dimSizeOFR;
        if (currentOutShape.isDynamic()) {
            dimSizeOFR = VPU::getDimValue(builder, operation, mcAxis);
        } else {
            dimSizeOFR = builder.getIndexAttr(currentOutShape[Dim(mcAxis)]);
        }

        auto balanced = computeBalancedTileValues(builder, loc, dimSizeOFR, numClusters, alignment[mcAxis]);
        if (!balanced) {
            return operation->emitError("Tiling algorithm failed");
        }

        log.trace("Balanced multiclustering: dim={0}, numClusters={1}, alignment={2}", Dim(mcAxis), numClusters,
                  alignment[mcAxis]);
        return applyBalancedMulticlusterTilingImpl(operation, builder, numClusters, Dim(mcAxis), balanced->smallTile,
                                                   balanced->numLarge, balanced->delta, log);
    }

    // E-162999 workaround: tileUsingSCF drops the output order in the ForAllOp and terminator.
    llvm::for_each(tilingResult->loops, [&](mlir::LoopLikeOpInterface loop) {
        fixForallOutputTypes(mlir::cast<mlir::scf::ForallOp>(loop.getOperation()), operation);
    });

    builder.replaceOp(operation, tilingResult->replacements);

    return mlir::success();
}

SmallVector<mlir::Operation*> MulticlusterTilingSCFAlgorithm::applySCFTilingAndFusion(mlir::Operation* /*operation*/,
                                                                                      mlir::RewriterBase& /*builder*/,
                                                                                      const MergeConfiguration&,
                                                                                      Logger log) {
    log.trace("MC fusion is not yet implemented.");
    return {};
}
