//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/Dialect/SCF/Transforms/TileUsingInterface.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Iterators.h>
#include <mlir/Interfaces/TilingInterface.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/CSE.h>
#include <mlir/Transforms/LoopInvariantCodeMotionUtils.h>
#include <mlir/Transforms/RegionUtils.h>

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLFunctionalExtras.h>

namespace vpux::ShaveCodeGen {
#define GEN_PASS_DECL_PARAMETRICCAPSULETILING
#define GEN_PASS_DEF_PARAMETRICCAPSULETILING

#include "vpux/compiler/ShaveCodeGen/passes.hpp.inc"
}  // namespace vpux::ShaveCodeGen

using namespace vpux;

namespace {

// This file implements a pass that does symbolical/parametric tile and fuse of ops in a capsule.
// The implementation is split into two parts: an analysis parts which determines subgraphs of ops
// which can fused together and possible tiling axes, and the transformation part which performs the actual
// tiling and fusion via tileConsumerAndFuseProducers. Later passes can outline the produced loops into
// separate kernels and extract the input slice computation logic. The resulting kernels can then become part
// of the regular tiling flow.

// ===========================================
//               Analysis
// ===========================================

mlir::FailureOr<mlir::AffineConstantExpr> getBoundedDim(mlir::Value value, int64_t dim) {
    auto shapedType = mlir::dyn_cast<mlir::ShapedType>(value.getType());
    if (!shapedType) {
        return mlir::failure();
    }
    if (shapedType.isDynamicDim(dim)) {
        // E#108588 - support dynamic shapes
        return mlir::failure();
    }
    return mlir::cast<mlir::AffineConstantExpr>(
            mlir::getAffineConstantExpr(shapedType.getDimSize(dim), value.getType().getContext()));
}

static int64_t getElementSizeInBits(mlir::Value value) {
    return static_cast<int64_t>(mlir::cast<mlir::ShapedType>(value.getType()).getElementType().getIntOrFloatBitWidth());
}

/// @brief Given a result produces an affine map from sizes of the that result to sizes
/// of the operations iteration space required to produce the result.
/// @param linalgOp The linalg op
/// @param resultNumber The result for which we're computing the map
/// @return mlir::failure() if failed, otherwise a affine map having the number of dimensions
/// equal to the rank of the output tensor to sizes in the iteration space
mlir::FailureOr<mlir::AffineMap> getIterationSpaceFromResult(mlir::linalg::LinalgOp linalgOp, int64_t resultNumber) {
    auto iterators = mlir::cast<mlir::TilingInterface>(linalgOp.getOperation()).getLoopIteratorTypes();
    // Determine the iteration space that produces this slice of the result.
    auto map = linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(resultNumber));

    SmallVector<mlir::AffineExpr> outputExprs(iterators.size(), mlir::getAffineConstantExpr(1, linalgOp.getContext()));
    for (auto res : llvm::enumerate(map.getResults())) {
        // Always casts because we're a projected permutation
        auto dimExpr = mlir::cast<mlir::AffineDimExpr>(res.value());
        outputExprs[dimExpr.getPosition()] = mlir::getAffineDimExpr(res.index(), linalgOp.getContext());
    }

    // Get bounded dims for all input dimensions and compute the non-sliced iteration space sizes.
    SmallVector<mlir::AffineExpr> sizeMapExprs;
    for (auto operand : linalgOp->getOperands()) {
        auto shapedType = mlir::dyn_cast<mlir::ShapedType>(operand.getType());
        if (!shapedType) {
            continue;
        }
        auto rank = shapedType.getRank();
        for (auto dim : llvm::seq<int64_t>(0, rank)) {
            auto boundedDim = getBoundedDim(operand, dim);
            if (mlir::failed(boundedDim)) {
                return mlir::failure();
            }
            sizeMapExprs.push_back(*boundedDim);
        }
    }

    // Create a map to the concatenated bounds of each operand
    // E.g. if we an op with two inputs of rank 3 and bonds [1, 10, 1], [2, 13, 2]
    // our size map will be:
    //   <d0, d1, d2, d3, d4, d5> -> <1, 10, 1, 2, 13, 2>
    auto sizeMap = mlir::AffineMap::get(/*dimCount=*/sizeMapExprs.size(), /*symbolCount=*/0, sizeMapExprs,
                                        linalgOp.getContext());

    // Get the shapes to loop map. As with the size map, this will take in
    // the concatenated input dimensions and produce the iteration space.
    // E.g. <d0, d1, d2, d3, d4, d5> -> <d0, d2, d3 + d5>.
    // We can compose that with the size map to get the actual bounds
    // of the dimensions of the iteration space that don't appear on the
    // projected permutation map of our result.
    //
    // We're guaranteed here to get a map, otherwise verification would fail.
    auto shapesToLoopMap = linalgOp.getShapesToLoopsMap();
    shapesToLoopMap = shapesToLoopMap.compose(sizeMap);

    auto unusedDims = mlir::getUnusedDimsBitVector(llvm::ArrayRef<mlir::AffineMap>{map});
    for (auto dim : llvm::seq<int64_t>(0, static_cast<int64_t>(unusedDims.size()))) {
        if (unusedDims.test(dim)) {
            outputExprs[dim] = shapesToLoopMap.getResult(dim);
        }
    }
    auto toInternal = mlir::AffineMap::get(/*dimCount=*/map.getResults().size(),
                                           /*symbolCount=*/0, outputExprs, linalgOp.getContext());
    return toInternal;
}

/// @brief Produces an affine map from the sizes of the output slice to the sizes of the input slices of the
/// fused producers.
/// @param op The tiling interface op to compute slice size bounds for.
/// @param resultNumber The result index whose output slice drives the input slice computation.
/// @return Per-operand affine maps from output slice sizes to input slice sizes, or failure.
mlir::FailureOr<SmallVector<mlir::AffineMap>> getFusedSliceSizeBounds(mlir::TilingInterface op, int64_t resultNumber) {
    auto iterators = op.getLoopIteratorTypes();

    if (auto linalgOp = mlir::dyn_cast<mlir::linalg::LinalgOp>(op.getOperation())) {
        auto iterationSpace = getIterationSpaceFromResult(linalgOp, resultNumber);
        if (mlir::failed(iterationSpace)) {
            return mlir::failure();
        }
        SmallVector<mlir::AffineMap> resultMaps;
        // For each operand we compose the map from output slice sizes to iteration space with the map
        // from iteration space to operand shape to get the map from output slice sizes to input slice sizes.
        for (auto& operand : linalgOp->getOpOperands()) {
            auto shapedType = mlir::dyn_cast<mlir::ShapedType>(operand.get().getType());
            if (!shapedType) {
                continue;
            }
            auto operandMap = linalgOp.getMatchingIndexingMap(&operand);
            auto composedMap = operandMap.compose(*iterationSpace);
            resultMaps.push_back(composedMap);
        }
        return resultMaps;
    }

    if (auto padOp = mlir::dyn_cast<mlir::tensor::PadOp>(op.getOperation())) {
        // The upper bounds of the input slice is exactly the identity map.
        auto rank = mlir::cast<mlir::RankedTensorType>(padOp->getResult(0).getType()).getRank();
        SmallVector<mlir::AffineMap> resultMaps(1, mlir::AffineMap::getMultiDimIdentityMap(rank, op->getContext()));
        return resultMaps;
    }

    return mlir::failure();
}

/// @brief Tracks slices of ops and estimates an upper bound for the memory requirements of the current subgraph.
///
/// Ops are expected to be added in reverse topological order (with the exception of the root op).
/// Slices are tracked as affine maps with dimensions being the slice sizes of the first result of the root op.
/// @note Since offsets are not currently tracked, slices cannot be de-duplicated; even if
/// two slices have the same affine maps they are assumed to be different. This can be improved in future
/// work by having a better approximation of DPS memory reuse and effects of in-kernel loop fusion (while still keeping
/// this as an upper bound).
class GraphStatsTracker {
private:
    struct CandidateInputSlices {
        mlir::AffineMap sizeMap;
        SmallVector<SmallVector<mlir::AffineMap>> inputSlices;
        int64_t numSlices = 0;
    };

public:
    GraphStatsTracker(int64_t maxSize): _rootOp(nullptr), _maxSize(maxSize) {
    }

    bool addRootOp(mlir::TilingInterface rootOp) {
        this->_rootOp = rootOp;
        auto firstResult = rootOp->getResult(0);
        auto inSliceBounds = getFusedSliceSizeBounds(rootOp, 0);
        if (mlir::failed(inSliceBounds)) {
            return false;
        }
        auto shapedType = mlir::dyn_cast<mlir::ShapedType>(firstResult.getType());
        if (!shapedType) {
            return false;
        }
        auto rank = shapedType.getRank();
        _sizeMap = mlir::AffineMap::get(rank, /*symbolCount=*/0, {mlir::getAffineConstantExpr(0, rootOp.getContext())},
                                        rootOp->getContext());

        // Construct the symbolic first result slice
        SmallVector<mlir::AffineExpr> rootExprs(rank, mlir::getAffineConstantExpr(1, rootOp.getContext()));
        for (int64_t dim = 0; dim < rank; ++dim) {
            if (!shapedType.isDynamicDim(dim) && shapedType.getDimSize(dim) == 1) {
                continue;
            }
            rootExprs[dim] = mlir::getAffineDimExpr(dim, rootOp.getContext());
        }
        auto rootSlice = mlir::AffineMap::get(rank, /*symbolCount=*/0, rootExprs, rootOp->getContext());
        _sizeMap = accumulateSliceSize(rootSlice, _sizeMap, getElementSizeInBits(firstResult));
        _numSlices++;
        _valueToSliceMap[firstResult] = {rootSlice};

        // Propagate slices from the first result.
        auto newInputSliceCandidates = getInputSlices(rootOp);
        if (mlir::failed(newInputSliceCandidates)) {
            return false;
        }
        commitCandidateInputSlices(*newInputSliceCandidates, rootOp);

        // Populate the other result slices.
        auto linalgOp = mlir::dyn_cast<mlir::linalg::LinalgOp>(rootOp.getOperation());
        if (rootOp->getNumResults() > 1) {
            if (!linalgOp) {
                return false;
            }
            auto iterationSpaceOrFailure = getIterationSpaceFromResult(linalgOp, 0);
            if (mlir::failed(iterationSpaceOrFailure)) {
                return false;
            }
            auto iterationSpace = *iterationSpaceOrFailure;
            for (int64_t i = 1; i < linalgOp->getNumResults(); ++i) {
                auto map = linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(i));
                map = map.compose(iterationSpace.compose(rootSlice));
                _valueToSliceMap[linalgOp->getResult(i)].push_back(map);
                _sizeMap = accumulateSliceSize(map, _sizeMap, getElementSizeInBits(linalgOp->getResult(i)));
                _numSlices++;
            }
        }

        auto profitableTilingAxes = isProfitableToFuseOp(_sizeMap);
        if (mlir::failed(profitableTilingAxes)) {
            return false;
        }
        _tilingDims = *profitableTilingAxes;

        return true;
    }

    bool tryAddingOp(mlir::TilingInterface op) {
        auto candidateInputSlices = getInputSlices(op);
        if (mlir::failed(candidateInputSlices)) {
            return false;
        }

        if (candidateInputSlices->numSlices + _numSlices > _maxSlices) {
            return false;
        }

        auto profitableTilingAxes = isProfitableToFuseOp(candidateInputSlices->sizeMap);
        if (mlir::failed(profitableTilingAxes)) {
            return false;
        }
        _tilingDims = *profitableTilingAxes;

        commitCandidateInputSlices(*candidateInputSlices, op);
        return true;
    }

    llvm::MapVector<int64_t, double>& getTilingDims() {
        return _tilingDims;
    }

    void printFormat(llvm::raw_ostream& stream) const {
        for (auto& entry : _valueToSliceMap) {
            printTo(stream, "Value: ");
            entry.first.printAsOperand(stream, mlir::OpPrintingFlags());
            printTo(stream, "\n");
            for (auto& map : entry.second) {
                printTo(stream, "   Slice {0}\n", map);
            }
        }
        printTo(stream, "Size: {0}\n", _sizeMap);
        printTo(stream, "Valid tiling dimensions:");
        for (auto& [dim, ratio] : _tilingDims) {
            printTo(stream, " {0} (ratio={1})", dim, ratio);
        }
    }

    void dump() const {
        printFormat(llvm::errs());
    }

private:
    /// @brief Determines whether tiling the current subgraph is profitable given the accumulated slice size map.
    /// @param newSizes Affine map representing the total memory footprint as a function of root op slice sizes.
    /// @return A map from tiling dimension index to a profitability score (ideally 1.0) for each valid tiling axis
    /// of the root op. If the subgraph already fits without tiling, all dimensions are returned
    /// with a score of 1.0. Otherwise, each dimension is evaluated by comparing the ratio of non-tiled to
    /// minimum-tiled memory footprint, and only dimensions exceeding minProfitableTilingRatio are kept.
    /// Returns failure if no profitable tiling axes exist or the subgraph cannot be made to fit.
    mlir::FailureOr<llvm::MapVector<int64_t, double>> isProfitableToFuseOp(mlir::AffineMap newSizes) {
        auto shapedType = mlir::dyn_cast<mlir::ShapedType>(_rootOp->getResult(0).getType());
        auto rank = shapedType.getRank();

        // Check if this fits into the cache without tiling, if so we can just fuse it.
        // We need to pass into the size calculation the bounds of the first result.
        SmallVector<mlir::AffineExpr> resultSizeExprs;
        for (int64_t i = 0; i < rank; ++i) {
            auto dim = getBoundedDim(_rootOp->getResult(0), i);
            if (mlir::failed(dim)) {
                return mlir::failure();
            }
            resultSizeExprs.push_back(*dim);
        }
        auto resultSizes = mlir::AffineMap::get(rank, 0, resultSizeExprs, _rootOp.getContext());
        auto nonTiledSize = newSizes.compose(resultSizes);

        auto nonTiledSizeExpr = mlir::dyn_cast<mlir::AffineConstantExpr>(nonTiledSize.getResult(0));
        if (!nonTiledSizeExpr) {
            return mlir::failure();
        }

        if (nonTiledSizeExpr.getValue() <= _maxSize) {
            llvm::MapVector<int64_t, double> allDims;
            for (int64_t dim = 0; dim < rank; ++dim) {
                auto boundedDim = getBoundedDim(_rootOp->getResult(0), dim);
                if (mlir::failed(boundedDim)) {
                    return mlir::failure();
                }
                if ((*boundedDim).getValue() <= _minDimTileSize) {
                    continue;
                }
                allDims[dim] = 1.0;
            }
            return allDims;
        }

        // Compute the profitability of tiling each dimension of the root op slice and collect the valid tiling
        // axes with their scores. Since each slice represents both computation and memory use, we can use the size
        // ratio between the tiled and non-tiled cases as a proxy for profitability. If we tile a dimension in x rougly
        // equal slices we expect to reduce the memory use/computation by roughly a factor of x to be profitable. We
        // compute a proxy score for this which should be as large as possible (ideally 1.0) and reject any dimensions
        // for which this score is not high enough.
        llvm::MapVector<int64_t, double> validTilingAxes;
        for (int64_t dim = 0; dim < rank; ++dim) {
            auto boundedDimOrFail = getBoundedDim(_rootOp->getResult(0), dim);
            if (mlir::failed(boundedDimOrFail)) {
                return mlir::failure();
            }
            auto boundedDim = (*boundedDimOrFail).getValue();
            if (boundedDim <= _minDimTileSize) {
                continue;
            }
            // Compute the minimum tile size
            SmallVector<mlir::AffineExpr> dimTileExprs;
            for (int64_t i = 0; i < rank; ++i) {
                if (i == dim) {
                    dimTileExprs.push_back(mlir::getAffineConstantExpr(_minDimTileSize, _rootOp.getContext()));
                    continue;
                }
                auto bounded = getBoundedDim(_rootOp->getResult(0), i);
                if (mlir::failed(bounded)) {
                    return mlir::failure();
                }
                dimTileExprs.push_back(*bounded);
            }
            auto minTileSizeMap = mlir::AffineMap::get(rank, 0, dimTileExprs, _rootOp.getContext());
            auto minTiledSize = newSizes.compose(minTileSizeMap);
            auto minTiledSizeExpr = mlir::dyn_cast<mlir::AffineConstantExpr>(minTiledSize.getResult(0));
            if (!minTiledSizeExpr) {
                continue;
            }

            // Compute a profitability score for tiling this dimension:
            //   ratio = (nonTiledMemory / dimSize) / (minTiledMemory / minTileSize)
            // A score of 1.0 means memory scales linearly with the tile size along this axis (ideal).
            // A lower score means other dimensions dominate memory, so tiling this axis is less profitable.
            llvm::APFloat nonTiledSizeAPF(llvm::APFloat::IEEEquad(),
                                          llvm::APInt(128, checked_cast<uint64_t>(nonTiledSizeExpr.getValue())));
            llvm::APFloat boundedDimAPF(llvm::APFloat::IEEEquad(),
                                        llvm::APInt(128, checked_cast<uint64_t>(boundedDim)));
            llvm::APFloat minDimTileSizeAPF(llvm::APFloat::IEEEquad(),
                                            llvm::APInt(128, checked_cast<uint64_t>(_minDimTileSize)));
            llvm::APFloat minTiledSizeAPF(llvm::APFloat::IEEEquad(),
                                          llvm::APInt(128, checked_cast<uint64_t>(minTiledSizeExpr.getValue())));

            llvm::APFloat ratioAPF = ((nonTiledSizeAPF / boundedDimAPF) * minDimTileSizeAPF) / minTiledSizeAPF;
            bool losesInfoConv = false;
            // Convert the APFloat to IEEEdouble, then we can do a convertToDouble to get the actual value.
            ratioAPF.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven, &losesInfoConv);
            double ratio = ratioAPF.convertToDouble();
            if (_minProfitableTilingRatio < ratio) {
                validTilingAxes[dim] = ratio;
            }
        }

        if (validTilingAxes.empty()) {
            // No valid tiling axes, fusion is not profitable
            return mlir::failure();
        }

        // We should be able to tile such that we fit into the cache.
        SmallVector<mlir::AffineExpr> minTileSizeExprs;
        for (int64_t dim = 0; dim < rank; ++dim) {
            if (validTilingAxes.count(dim)) {
                minTileSizeExprs.push_back(mlir::getAffineConstantExpr(_minDimTileSize, _rootOp.getContext()));
                continue;
            }
            auto boundedDim = getBoundedDim(_rootOp->getResult(0), dim);
            if (mlir::failed(boundedDim)) {
                return mlir::failure();
            }
            minTileSizeExprs.push_back(*boundedDim);
        }
        auto minTileSizeMap = mlir::AffineMap::get(rank, 0, minTileSizeExprs, _rootOp.getContext());
        minTileSizeMap = newSizes.compose(minTileSizeMap);
        auto minTiledSizeExpr = mlir::dyn_cast<mlir::AffineConstantExpr>(minTileSizeMap.getResult(0));
        if (!minTiledSizeExpr) {
            return mlir::failure();
        }
        if (minTiledSizeExpr.getValue() > _maxSize) {
            return mlir::failure();
        }

        return validTilingAxes;
    }

    mlir::FailureOr<CandidateInputSlices> getInputSlices(mlir::TilingInterface op) {
        CandidateInputSlices result = {_sizeMap, {}, 0};
        result.inputSlices.append(llvm::count_if(op->getOperands(),
                                                 [](mlir::Value operand) {
                                                     return mlir::isa<mlir::TensorType>(operand.getType());
                                                 }),
                                  {});

        for (auto res : op->getResults() | indexed) {
            auto inSliceBounds = getFusedSliceSizeBounds(op, res.index());
            if (mlir::failed(inSliceBounds)) {
                return mlir::failure();
            }
            int64_t tensorCount = 0;

            for (auto operand : op->getOperands()) {
                if (!mlir::isa<mlir::TensorType>(operand.getType())) {
                    // Skip non-tensor operands
                    continue;
                }
                if (operand.getDefiningOp() && mlir::isa<mlir::tensor::EmptyOp>(operand.getDefiningOp())) {
                    // Add a zero-size slice for tensor.empty since this would be tied to an output
                    // anyway and this avoid double counting.
                    auto emptySliceMap = (*inSliceBounds)[tensorCount];
                    SmallVector<mlir::AffineExpr> zeroResults(emptySliceMap.getNumResults(),
                                                              mlir::getAffineConstantExpr(0, op.getContext()));
                    result.inputSlices[tensorCount].push_back(
                            mlir::AffineMap::get(emptySliceMap.getNumDims(), 0, zeroResults, op.getContext()));
                    ++tensorCount;
                    continue;
                }
                auto am = (*inSliceBounds)[tensorCount];
                for (auto& map : _valueToSliceMap[res.value()]) {
                    auto composedMap = am.compose(map);
                    result.sizeMap = accumulateSliceSize(composedMap, result.sizeMap, getElementSizeInBits(operand));
                    result.numSlices++;
                    result.inputSlices[tensorCount].push_back(composedMap);
                }
                ++tensorCount;
            }
        }
        return result;
    }

    void commitCandidateInputSlices(CandidateInputSlices& candidates, mlir::TilingInterface op) {
        int64_t tensorCount = 0;
        for (auto operand : op->getOperands()) {
            if (!mlir::isa<mlir::TensorType>(operand.getType())) {
                // Skip non-tensor operands
                continue;
            }

            _valueToSliceMap[operand].append(candidates.inputSlices[tensorCount].begin(),
                                             candidates.inputSlices[tensorCount].end());
            tensorCount++;
        }
        _sizeMap = candidates.sizeMap;
        _numSlices += candidates.numSlices;
    }

    mlir::AffineMap accumulateSliceSize(mlir::AffineMap map, mlir::AffineMap runningSize, int64_t typeSizeInBits) {
        auto* ctx = map.getContext();
        mlir::AffineExpr product = mlir::getAffineConstantExpr(typeSizeInBits, ctx);
        for (auto expr : map.getResults()) {
            product = product * expr;
        }
        auto newExpr = runningSize.getResult(0) + product;
        return mlir::AffineMap::get(map.getNumDims(), map.getNumSymbols(), {newExpr}, ctx);
    }

    mlir::TilingInterface _rootOp;
    // The size map is an affine map from the slice sizes of the root op to and expression of
    // the total size in bits of all slices in the current subgraph (the map has a single result).
    mlir::AffineMap _sizeMap;
    // Maximum size of the memory in bits.
    int64_t _maxSize;
    // For each value we track the affine maps representing the slices of that value
    // that are used in the current subgraph. Note that we can have multiple slices for the
    // same value.
    llvm::DenseMap<mlir::Value, SmallVector<mlir::AffineMap>> _valueToSliceMap;
    // The current valid tiling dimensions and their profitability scores.
    llvm::MapVector<int64_t, double> _tilingDims;
    // Total number of slices in the current graph.
    int _numSlices = 0;
    // Limit the number of tracked slices in order to avoid degenerate
    // cases that could blow up compile time.
    static int constexpr _maxSlices = 100;
    static int64_t constexpr _minDimTileSize = 1;
    static double constexpr _minProfitableTilingRatio = 0.9;
};

struct TilingAnalysisResult {
    llvm::MapVector<int64_t, double> tilingAxes;
    llvm::SmallSetVector<mlir::TilingInterface, 8> opsToFuse;
};

/// @brief Collects a maximal set of producer ops that can be profitably fused with @p rootOp under the
/// CMX memory constraint @p cmxSize.
///
/// Starting from @p rootOp, walks its operands in reverse topological order and greedily adds producers
/// that satisfy @p isProducerTileable and whose all users are already in the fused set.
///
/// @param rootOp The consumer op that anchors the fusion subgraph.
/// @param isProducerTileable Predicate that returns true for ops eligible for fusion.
/// @param cmxSize Available CMX memory budget.
/// @return The set of fusible ops together with the profitable tiling axes and their scores,
/// or failure if the root op itself cannot be tiled.
static mlir::FailureOr<TilingAnalysisResult> collectFusibleProducerOps(
        mlir::TilingInterface rootOp, llvm::function_ref<bool(mlir::Operation*)> isProducerTileable, const Bit cmxSize,
        Logger& log) {
    llvm::SmallSetVector<mlir::TilingInterface, 8> opsToFuse;
    std::set<mlir::TilingInterface> opsToVisit;

    log.trace("Collecting fusible producer ops for root operation {0} ", rootOp);

    GraphStatsTracker tracker(cmxSize.count());
    if (!tracker.addRootOp(rootOp)) {
        log.trace("Root operation {0} cannot be added to the graph", rootOp);
        return mlir::failure();
    }

    auto* rootBlock = rootOp->getBlock();
    auto* rootOpPtr = rootOp.getOperation();
    auto addOperands = [&](mlir::TilingInterface op) {
        for (mlir::Value operand : op->getOperands()) {
            auto* producer = operand.getDefiningOp();

            if (producer == nullptr || producer->getBlock() != rootBlock || !isProducerTileable(producer)) {
                continue;
            }

            auto producerTileable = mlir::cast<mlir::TilingInterface>(producer);
            opsToVisit.insert(producerTileable);
        }
    };
    addOperands(rootOp);
    opsToFuse.insert(rootOp);
    if (opsToVisit.empty()) {
        return TilingAnalysisResult{std::move(tracker.getTilingDims()), std::move(opsToFuse)};
    }

    for (auto currentIt = rootOpPtr->getIterator(); currentIt != rootBlock->begin() && !opsToVisit.empty();) {
        --currentIt;
        mlir::TilingInterface currentOp = mlir::dyn_cast<mlir::TilingInterface>(&*currentIt);

        if (!currentOp || !opsToVisit.count(currentOp)) {
            continue;
        }
        opsToVisit.erase(currentOp);

        // All users should be in the current fused set, otherwise we end up
        // duplicating compute. This may be profitable to fuse but unlikely so we skip it for now.
        // A counterexample to this would be a linalg.fill feeding into anything.
        if (!llvm::all_of(currentOp->getUsers(), [&](mlir::Operation* user) {
                auto tileableUser = mlir::dyn_cast<mlir::TilingInterface>(user);
                return tileableUser && opsToFuse.contains(tileableUser);
            })) {
            log.trace("Operation {0} has users outside the fusion set, skipping", currentOp);
            continue;
        }

        if (!tracker.tryAddingOp(currentOp)) {
            // We can't add this op.
            log.trace("Operation {0} cannot be added to the fusion set, skipping", currentOp);
            continue;
        }

        opsToFuse.insert(currentOp);
        addOperands(currentOp);
    }

    log.trace("Graph state:\n{0}", tracker);

    return TilingAnalysisResult{std::move(tracker.getTilingDims()), std::move(opsToFuse)};
}

static bool hasOffsetIndependentSliceSizes(mlir::AffineMap operandMap) {
    auto* ctx = operandMap.getContext();
    const auto numDims = operandMap.getNumDims();
    if (operandMap.getNumSymbols() != 0) {
        return false;
    }

    // It is enough to check that if for an operand we have:
    //     (d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 + s0, d1 + s1, d2 + s2, d3 + s3)
    // and the operand affine map:
    //     (d0, d1, d2, d3) -> AM(d0, d1, d2, d3)
    // then:
    //     (d0, d1, d2, d3)[s0, s1, s2, s3]
    //         -> AM(d0 + s0, d1 + s1, d2 + s2, d3 + s3) - AM(d0, d1, d2, d3)
    // does not depend on d0, d1, d2, d3.
    const auto identityDimMap = mlir::AffineMap::getMultiDimIdentityMap(numDims, ctx);

    SmallVector<mlir::AffineExpr> shiftedDimReplacements;
    shiftedDimReplacements.reserve(numDims);
    for (unsigned dim = 0; dim < numDims; ++dim) {
        auto dimExpr = mlir::getAffineDimExpr(dim, ctx);
        shiftedDimReplacements.push_back(dimExpr + mlir::getAffineSymbolExpr(dim, ctx));
    }

    const auto shiftedDimsMap = mlir::AffineMap::get(numDims, numDims, shiftedDimReplacements, ctx);
    const auto baseDimsMap = mlir::AffineMap::get(numDims, numDims, identityDimMap.getResults(), ctx);

    const auto shiftedOperandMap = operandMap.compose(shiftedDimsMap);
    const auto baseOperandMap = operandMap.compose(baseDimsMap);

    const auto numExtendedSymbols = shiftedOperandMap.getNumSymbols();
    for (unsigned resultIdx = 0; resultIdx < shiftedOperandMap.getNumResults(); ++resultIdx) {
        auto deltaExpr =
                mlir::simplifyAffineExpr(shiftedOperandMap.getResult(resultIdx) - baseOperandMap.getResult(resultIdx),
                                         numDims, numExtendedSymbols);
        for (unsigned dim = 0; dim < numDims; ++dim) {
            if (deltaExpr.isFunctionOfDim(dim)) {
                return false;
            }
        }
    }

    return true;
}

static bool isTileableLinalg(mlir::linalg::LinalgOp linalgOp) {
    // Check that the output maps are projected permutations. This
    // constraint comes from the linalg tiling model.
    for (int64_t i : irange(linalgOp.getNumDpsInits())) {
        auto init = linalgOp.getDpsInitOperand(i);
        auto map = linalgOp.getMatchingIndexingMap(init);
        if (!map.isProjectedPermutation(/*allowZero=*/false)) {
            return false;
        }
    }
    // Reject degenerate cases where the op is using non-parameter tensors
    // (see the output from the mlir convert-conv2d-to-img2col.mlir test).
    if (auto genericOp = mlir::dyn_cast<mlir::linalg::GenericOp>(linalgOp.getOperation())) {
        llvm::SetVector<mlir::Value> usedAbove;
        mlir::getUsedValuesDefinedAbove(genericOp.getRegion(), usedAbove);
        for (auto value : usedAbove) {
            if (mlir::isa<mlir::TensorType>(value.getType())) {
                return false;
            }
        }
    }

    for (auto& operand : linalgOp->getOpOperands()) {
        auto shapedType = mlir::dyn_cast<mlir::ShapedType>(operand.get().getType());
        if (!shapedType) {
            continue;
        }

        auto operandMap = linalgOp.getMatchingIndexingMap(&operand);
        // Check for offset-independent slice sizes. This will essentially reject
        // any expressions containg floordiv, ceildiv and mod. Alternatively we could
        // compute the upper bound of an affine expression, but would still require that
        // all expressions in the map are pure affine.
        // This will currently capture all of our ops and we don't have any examples
        // of ops that would need anything more complex.
        if (!hasOffsetIndependentSliceSizes(operandMap)) {
            return false;
        }
    }

    return true;
}

static void collectTileableOps(IE::CodeGenCapsuleOp capsule, llvm::SmallSetVector<mlir::Operation*, 8>& tileableOps) {
    capsule->walk([&](mlir::TilingInterface op) {
        if (auto linalgOp = mlir::dyn_cast<mlir::linalg::LinalgOp>(op.getOperation())) {
            if (!isTileableLinalg(linalgOp)) {
                return;
            }
        }
        if (auto linalgOp = mlir::dyn_cast<mlir::linalg::SoftmaxOp>(op.getOperation())) {
            // linalg.softmax can't currently be fused into a consumer due to missing
            // the generateResultTileValue implementation.
            return;
        }
        tileableOps.insert(op.getOperation());
    });
}

// ===========================================
//               Transformation
// ===========================================

//
// ParametricCapsuleTilingPass
//

class ParametricCapsuleTilingPass final :
        public ShaveCodeGen::impl::ParametricCapsuleTilingBase<ParametricCapsuleTilingPass> {
public:
    explicit ParametricCapsuleTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void runOnCapsule(IE::CodeGenCapsuleOp capsule);
    mlir::FailureOr<mlir::scf::ForOp> tileOp(mlir::TilingInterface op,
                                             llvm::SmallSetVector<mlir::Operation*, 8>& tileableOps, int64_t loopId,
                                             mlir::IRRewriter& rewriter, Bit cmxSize);
    mlir::FailureOr<TilingAnalysisResult> analyzeTilingOp(mlir::TilingInterface op,
                                                          const llvm::SmallSetVector<mlir::Operation*, 8>& tileableOps,
                                                          Bit cmxSize);
    void safeRunOnFunc() final;
};

void ParametricCapsuleTilingPass::runOnCapsule(IE::CodeGenCapsuleOp capsule) {
    int64_t loopId = 0;
    const Bit cmxSize = VPU::getTotalCMXSize(capsule).to<Bit>();

    llvm::SmallSetVector<mlir::Operation*, 8> tileableOps;
    collectTileableOps(capsule, tileableOps);

    mlir::IRRewriter rewriter(capsule);

    llvm::SmallSetVector<mlir::Operation*, 4> tiledOps;
    capsule->walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>([&](mlir::Operation* op) {
        if (tiledOps.contains(op)) {
            return mlir::WalkResult::skip();
        }
        if (mlir::isOpTriviallyDead(op)) {
            // We are safe to remove the op here since the walk uses an make_early_inc_range
            // to iterate through the block.
            tileableOps.remove(op);
            rewriter.eraseOp(op);
            return mlir::WalkResult::advance();
        }
        if (auto tileableOp = mlir::dyn_cast<mlir::TilingInterface>(op)) {
            if (!tileableOps.contains(op)) {
                return mlir::WalkResult::advance();
            }
            auto loop = tileOp(tileableOp, tileableOps, loopId, rewriter, cmxSize);
            if (mlir::failed(loop)) {
                return mlir::WalkResult::advance();
            }
            tileableOps.remove(op);
            tiledOps.insert(loop->getOperation());
            loopId++;
        }
        return mlir::WalkResult::advance();
    });
}

mlir::FailureOr<mlir::scf::ForOp> ParametricCapsuleTilingPass::tileOp(
        mlir::TilingInterface op, llvm::SmallSetVector<mlir::Operation*, 8>& tileableOps, int64_t loopId,
        mlir::IRRewriter& rewriter, Bit cmxSize) {
    auto analysisResult = analyzeTilingOp(op, tileableOps, cmxSize);
    if (mlir::failed(analysisResult)) {
        _log.trace("Could not analyze operation {0} for tiling, skipping", op);
        return mlir::failure();
    }

    if (analysisResult->tilingAxes.empty()) {
        _log.trace("Sub-graph is valid with no available tiling dimensions, marking entire sub-graph as non-tileable");
        for (auto toFuseOp : analysisResult->opsToFuse) {
            tileableOps.remove(toFuseOp);
        }
        return mlir::failure();
    }

    _log.trace("Tiling operation {0}", op);

    rewriter.setInsertionPointAfter(op);

    // Collect ops created during tiling so they can be cleaned up on any failure path.
    llvm::SmallVector<mlir::Operation*> opsToCleanUp;
    auto failureCleanup = [&]() {
        for (auto* createdOp : llvm::reverse(opsToCleanUp)) {
            if (createdOp && !createdOp->use_empty()) {
                continue;
            }
            rewriter.eraseOp(createdOp);
        }
    };

    auto loopIdCst = rewriter.create<mlir::arith::ConstantIndexOp>(op->getLoc(), loopId);
    opsToCleanUp.push_back(loopIdCst);

    // Create a tiling root op that will be used for parametric tiling
    SmallVector<mlir::Value> tilingRootOperands;
    for (auto result : op->getResults()) {
        tilingRootOperands.push_back(result);
    }

    SmallVector<mlir::Attribute> tilingRanksVec;
    for (auto [rank, _] : analysisResult->tilingAxes) {
        tilingRanksVec.push_back(rewriter.getI64IntegerAttr(rank));
    }
    auto tilingRanksAttr = mlir::ArrayAttr::get(rewriter.getContext(), tilingRanksVec);
    auto tilingRootOp = rewriter.create<Shave::TilingRootOp>(op->getLoc(), op->getResultTypes(), loopIdCst,
                                                             tilingRootOperands, tilingRanksAttr);
    opsToCleanUp.push_back(tilingRootOp);

    // Tile tilingRootOp using mlir::tileConsumerAndFuseProducers with tile size 1
    SmallVector<mlir::OpFoldResult> tileSizes(1, rewriter.getIndexAttr(1));
    mlir::scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes);
    tilingOptions.setLoopType(mlir::scf::SCFTilingOptions::LoopType::ForOp);

    mlir::scf::SCFTileAndFuseOptions tileAndFuseOptions;
    tileAndFuseOptions.tilingOptions = std::move(tilingOptions);
    mlir::scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
            [&](mlir::tensor::ExtractSliceOp, mlir::OpResult originalProducer,
                bool) -> std::optional<mlir::scf::SCFTileAndFuseOptions::ControlFnResult> {
        auto tileOp = mlir::dyn_cast<mlir::TilingInterface>(originalProducer.getOwner());
        if (!tileOp) {
            return std::nullopt;
        }
        if (analysisResult->opsToFuse.contains(tileOp)) {
            return mlir::scf::SCFTileAndFuseOptions::ControlFnResult{};
        }
        return std::nullopt;
    };
    tileAndFuseOptions.setFusionControlFn(std::move(controlFn));

    mlir::RewritePatternSet patterns(op.getContext());
    mlir::tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, op.getContext());
    mlir::tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    mlir::tensor::populateBubbleUpExtractSliceOpPatterns(patterns);
    mlir::tensor::populateFoldTensorEmptyPatterns(patterns);
    // Handle tensor.pad operations via cleanup patterns with no zero slice guard
    // to avoid any control flow being generated from tensor.pad tiling.
    llvm::SmallSetVector<mlir::TilingInterface, 8> fusedPadOps;
    patterns.insert<mlir::linalg::ExtractSliceOfPadTensorSwapPattern>(
            op.getContext(), [&](mlir::tensor::ExtractSliceOp extractSlice) -> std::optional<bool> {
                auto tileOp = mlir::dyn_cast_or_null<mlir::TilingInterface>(extractSlice.getOperand(0).getDefiningOp());
                if (tileOp && analysisResult->opsToFuse.contains(tileOp)) {
                    fusedPadOps.insert(tileOp);
                    return false;
                }
                return std::nullopt;
            });
    tileAndFuseOptions.cleanupPatterns = std::move(patterns);

    auto tilingResult = mlir::scf::tileConsumerAndFuseProducersUsingSCF(
            rewriter, mlir::cast<mlir::TilingInterface>(tilingRootOp.getOperation()), tileAndFuseOptions);
    if (failed(tilingResult)) {
        _log.trace("Failed to tile operation {0}", op);
        failureCleanup();
        return mlir::failure();
    }
    if (tilingResult->loops.empty()) {
        _log.trace("Tiling operation {0} produced no loops", op);
        failureCleanup();
        return mlir::failure();
    }

    auto loop = mlir::cast<mlir::scf::ForOp>(tilingResult->loops.front());
    opsToCleanUp.push_back(loop);
    if (!fusedPadOps.contains(op) && !tilingResult->fusedProducers.contains(op)) {
        // We didn't manage to fuse the original op, do some cleanup and return failure.
        _log.trace("Could not tile root operation {0}", op);
        failureCleanup();
        return mlir::failure();
    }

    // Remove any dead extract slice ops, otherwise they will keep alive ops which
    // really should be dead and we'll end up doing extra work.
    if (failed(mlir::runRegionDCE(rewriter, loop.getRegion()))) {
        _log.trace("Failed to run DCE in loop after tiling operation {0}", op);
        failureCleanup();
        return mlir::failure();
    }

    mlir::DominanceInfo domInfo;
    mlir::eliminateCommonSubExpressions(rewriter, domInfo, loop);
    mlir::moveLoopInvariantCode(loop);

    for (auto res : tilingRootOp->getResults() | indexed) {
        if (auto replacement = tilingResult->replacements.lookup(res.value())) {
            rewriter.replaceAllUsesWith(res.value(), replacement);
            rewriter.replaceAllUsesWith(op->getResult(res.index()), replacement);
        }
    }
    _log.trace("Successfully tiled operation {0}", op);
    rewriter.eraseOp(tilingRootOp);
    rewriter.eraseOp(op);

    return loop;
}

mlir::FailureOr<TilingAnalysisResult> ParametricCapsuleTilingPass::analyzeTilingOp(
        mlir::TilingInterface op, const llvm::SmallSetVector<mlir::Operation*, 8>& tileableOps, Bit cmxSize) {
    if (op->getNumResults() > 1) {
        // We can't yet handle more than one result.
        _log.trace("Operation {0} produces more than one result, skipping", op);
        return mlir::failure();
    }
    if (!tileableOps.contains(op.getOperation())) {
        _log.trace("Operation {0} is not tileable, skipping", op);
        return mlir::failure();
    }
    return collectFusibleProducerOps(
            op,
            [&](mlir::Operation* producer) {
                return tileableOps.contains(producer);
            },
            cmxSize, _log);
}

void ParametricCapsuleTilingPass::safeRunOnFunc() {
    auto func = getOperation();
    for (auto capsule : func.getOps<IE::CodeGenCapsuleOp>()) {
        runOnCapsule(capsule);
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::ShaveCodeGen::createParametricCapsuleTilingPass(Logger log) {
    return std::make_unique<ParametricCapsuleTilingPass>(log);
}
