//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement_fwd.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/scf/scf_analysis_utils.hpp"

#include <llvm/ADT/DenseMap.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

#include <optional>

namespace vpux::VPU {

/**
 * Baseline fully static shape of a type: bounds are substitutes only for dynamic dimensions;
 * a static dimension remains authoritative even when its carried bound is stale or less
 * precise. For example, tensor<1x?x11x16, bounds=[1, 48, 64, 16]> resolves to [1, 48, 11, 16],
 * not [1, 48, 64, 16]. Such types appear when SCF tiling makes a dimension static while
 * preserving bounds propagated from the untiled tensor.
 */
Shape getShapeUsingBoundsForDynamicDims(mlir::Type type);

/**
 * The insertion op an NCE result is written into, plus the shape-changing ops sitting between
 * the two. The tile shape of the NCE op is recovered by back-inferring the inserted slice
 * through `shapeTransforms` in reverse.
 */
struct InsertionPath {
    mlir::OffsetSizeAndStrideOpInterface insertionOp;
    SmallVector<VPU::TilingBuilderOpInterface> shapeTransforms;
};

/**
 * Find the insertion op (OffsetSizeAndStrideOpInterface) that consumes the NCE result,
 * recording the shape-changing ops that must be back-inferred from the inserted tile.
 * Returns std::nullopt if not found.
 *
 * Uses DestinationStyleOpInterface to identify insert-like ops (covers tensor.insert_slice).
 * tensor.parallel_insert_slice is also an insertion op but lives in the forall terminator
 * region and does not implement DestinationStyleOpInterface.
 */
std::optional<InsertionPath> findInsertionPath(VPU::NCEOpInterface nceOp);

/**
 * Resolves the concrete shape a value takes for one fixed iteration of the enclosing
 * scf.for / scf.forall loop nest.
 *
 * The resolver walks the producer chain backwards and understands the op kinds the SCF
 * tiling and Vertical Fusion infrastructure emit inside the loop body:
 *
 *   tensor.extract_slice   -> the evaluated mixed sizes are the result shape
 *   tensor.pad             -> source shape + low + high padding (all evaluated per iteration)
 *   VPU.Expand             -> source shape + static pads_begin + pads_end
 *   VPU.Slice              -> static_sizes, dynamic entries taken from the source shape
 *   tensor.cast / VPU.Copy / VPU.QuantizeCast -> shape of the source
 *   VPU.NCE.*              -> output shape of that NCE op (recursive, see getNceOutputShape),
 *                             taken from the insertion op it feeds and back-inferred through any
 *                             shape transform on the way there (see findInsertionPath)
 *
 * The recursion through VPU.NCE.* is what makes Vertical Fusion chains work. In a VF
 * region only the LAST NCE op of the chain feeds an insert_slice; every intermediate op
 * feeds `tensor.pad -> next NCE`. Without back-tracking through the chain, the per-iteration
 * shape of an intermediate op cannot be derived from an insertion op and the caller would
 * fall back to the (much larger) bounded shape, describing the whole tensor instead of the
 * actual tile.
 *
 * A resolver instance is bound to one iteration: `ivMap` must pin every induction variable of
 * the enclosing loop nest to a single value. Resolved shapes are cached per instance, so one
 * instance should be created per iteration and reused for all the values of that iteration.
 */
class IterationShapeResolver {
public:
    IterationShapeResolver(OpChainAnalysis& analysis, ValueRangeMap& ivMap, Logger log)
            : _analysis(analysis), _ivMap(ivMap), _log(log) {
    }

    /// Shape of `value` for the current iteration, or nullopt when it cannot be derived.
    std::optional<Shape> getValueShape(mlir::Value value, int depth = 0);

    /// Output shape of `nceOp` for the current iteration, or nullopt when it cannot be derived.
    std::optional<Shape> getNceOutputShape(VPU::NCEOpInterface nceOp, int depth = 0);

private:
    static constexpr int MAX_CHAIN_DEPTH = 32;

    std::optional<int64_t> eval(mlir::OpFoldResult ofr);
    std::optional<Shape> resolveSlice(mlir::OffsetSizeAndStrideOpInterface sliceIface, const Shape& baseShape);
    std::optional<Shape> resolveInsertionPath(const InsertionPath& path, const Shape& baseShape);
    std::optional<Shape> resolvePad(mlir::tensor::PadOp padOp, const Shape& baseShape, int depth);
    std::optional<Shape> resolveExpand(VPU::ExpandOp expandOp, const Shape& baseShape, int depth);
    std::optional<Shape> resolveVpuSlice(VPU::SliceOp sliceOp, const Shape& baseShape, int depth);
    std::optional<Shape> inferOutputFromInput(VPU::NCEOpInterface nceOp, const Shape& baseShape, int depth);

    OpChainAnalysis& _analysis;
    ValueRangeMap& _ivMap;
    llvm::DenseMap<mlir::Value, Shape> _cache;
    Logger _log;
};

}  // namespace vpux::VPU
