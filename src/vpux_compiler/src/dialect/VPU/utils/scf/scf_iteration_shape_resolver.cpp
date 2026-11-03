//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scf/scf_iteration_shape_resolver.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Interfaces/DestinationStyleOpInterface.h>

using namespace vpux;
using namespace VPU;

namespace vpux::VPU {

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

std::optional<InsertionPath> findInsertionPath(VPU::NCEOpInterface nceOp) {
    // If the NCE result has multiple users, the insertion op is ambiguous (could lead to
    // non-deterministic shape enumeration). Let the caller fall back to bounded-shape handling.
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
    //   pass-through    = tensor.cast | VPU.Copy | VPU.NCEEltwise  (shape-preserving, skipped silently)
    //   shape-transform = VPU.DepthToSpace                         (recorded in shapeTransforms so the
    //                                                               NCE tile shape can be back-inferred
    //                                                               from the inserted slice)
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

std::optional<Shape> IterationShapeResolver::getValueShape(mlir::Value value, int depth) {
    if (depth > MAX_CHAIN_DEPTH) {
        return std::nullopt;
    }
    auto ndType = mlir::dyn_cast<vpux::NDTypeInterface>(value.getType());
    if (ndType == nullptr) {
        return std::nullopt;
    }

    const auto declaredShape = ndType.getShape();
    Shape baseShape = getShapeUsingBoundsForDynamicDims(value.getType());
    if (!declaredShape.isDynamic()) {
        return baseShape;
    }

    const auto cached = _cache.find(value);
    if (cached != _cache.end()) {
        return cached->second;
    }

    auto* defOp = value.getDefiningOp();
    if (defOp == nullptr) {
        return std::nullopt;
    }

    std::optional<Shape> resolved;
    if (mlir::isa<mlir::tensor::CastOp, VPU::CopyOp, VPU::QuantizeCastOp>(defOp)) {
        resolved = getValueShape(defOp->getOperand(0), depth + 1);
    } else if (auto padOp = mlir::dyn_cast<mlir::tensor::PadOp>(defOp)) {
        resolved = resolvePad(padOp, baseShape, depth);
    } else if (auto expandOp = mlir::dyn_cast<VPU::ExpandOp>(defOp)) {
        resolved = resolveExpand(expandOp, baseShape, depth);
    } else if (auto vpuSliceOp = mlir::dyn_cast<VPU::SliceOp>(defOp)) {
        resolved = resolveVpuSlice(vpuSliceOp, baseShape, depth);
    } else if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(defOp)) {
        resolved = getNceOutputShape(nceOp, depth + 1);
    } else if (auto sliceIface = mlir::dyn_cast<mlir::OffsetSizeAndStrideOpInterface>(defOp)) {
        resolved = resolveSlice(sliceIface, baseShape);
    }

    if (!resolved.has_value() || resolved->size() != baseShape.size()) {
        return std::nullopt;
    }
    // The declared static dimensions are authoritative; keep them and use the resolved
    // values only for the dynamic ones.
    auto result = std::move(resolved.value());
    for (size_t i = 0; i < baseShape.size(); ++i) {
        if (declaredShape[Dim(i)] != mlir::ShapedType::kDynamic) {
            result[Dim(i)] = declaredShape[Dim(i)];
        }
    }
    _cache.try_emplace(value, result);
    return result;
}

std::optional<Shape> IterationShapeResolver::getNceOutputShape(VPU::NCEOpInterface nceOp, int depth) {
    if (depth > MAX_CHAIN_DEPTH) {
        return std::nullopt;
    }
    auto outValue = nceOp->getResult(0);
    auto outNdType = mlir::cast<vpux::NDTypeInterface>(outValue.getType());
    const auto declaredShape = outNdType.getShape();
    Shape baseShape = getShapeUsingBoundsForDynamicDims(outValue.getType());
    if (!declaredShape.isDynamic()) {
        return baseShape;
    }

    const auto cached = _cache.find(outValue);
    if (cached != _cache.end()) {
        return cached->second;
    }

    std::optional<Shape> resolved;
    // Preferred source of truth: the insertion op the result is written into. Available
    // for a standalone tiled NCE op and for the last op of a Vertical Fusion chain.
    if (const auto insertionPath = findInsertionPath(nceOp)) {
        resolved = resolveInsertionPath(insertionPath.value(), baseShape);
    }
    // Otherwise (intermediate op of a Vertical Fusion chain) forward-infer the output
    // shape from the resolved input shape and the op's own kernel / strides / pads.
    if (!resolved.has_value()) {
        resolved = inferOutputFromInput(nceOp, baseShape, depth);
    }

    if (!resolved.has_value() || resolved->size() != baseShape.size()) {
        return std::nullopt;
    }
    auto result = std::move(resolved.value());
    for (size_t i = 0; i < baseShape.size(); ++i) {
        if (declaredShape[Dim(i)] != mlir::ShapedType::kDynamic) {
            // Mismatch between the inferred value and the declared static dimension means
            // the chain was mis-modelled. Bail out instead of emitting bogus workloads.
            if (result[Dim(i)] != declaredShape[Dim(i)]) {
                _log.trace("Inferred dim {0} = {1} contradicts the declared static size {2} for op at '{3}'", i,
                           result[Dim(i)], declaredShape[Dim(i)], nceOp->getLoc());
                return std::nullopt;
            }
        }
    }
    _cache.try_emplace(outValue, result);
    return result;
}

std::optional<int64_t> IterationShapeResolver::eval(mlir::OpFoldResult ofr) {
    auto values = _analysis.getOpFoldResultValue(ofr, _ivMap, OpChainAnalysis::MODE::ALL_VALUES);
    if (!values.has_value() || values->empty()) {
        return std::nullopt;
    }
    // `_ivMap` pins every induction variable to a single value, so a well-modelled
    // expression must evaluate to exactly one result. More than one means the expression
    // depends on something outside the map and the iteration is not fully determined;
    // picking an arbitrary element would silently produce a plausible but wrong shape.
    if (values->size() != 1) {
        _log.trace("OpFoldResult evaluated to {0} values for a fully pinned iteration; cannot resolve shape",
                   values->size());
        return std::nullopt;
    }
    return values->front();
}

std::optional<Shape> IterationShapeResolver::resolveSlice(mlir::OffsetSizeAndStrideOpInterface sliceIface,
                                                          const Shape& baseShape) {
    auto mixedSizes = sliceIface.getMixedSizes();
    if (mixedSizes.size() != baseShape.size()) {
        // Rank-reducing slices are not modelled.
        return std::nullopt;
    }
    Shape result = baseShape;
    for (size_t i = 0; i < mixedSizes.size(); ++i) {
        auto size = eval(mixedSizes[i]);
        if (!size.has_value() || *size <= 0) {
            return std::nullopt;
        }
        result[Dim(i)] = *size;
    }
    return result;
}

/**
 * Shape of the NCE result that produced the tile written by `path.insertionOp`.
 *
 * The inserted slice describes the shape *after* the recorded shape transforms, so it is
 * back-inferred through them in reverse to land on the NCE op's own output shape. With no
 * transforms recorded this is just the evaluated slice.
 */
std::optional<Shape> IterationShapeResolver::resolveInsertionPath(const InsertionPath& path, const Shape& baseShape) {
    auto insertedShape = resolveSlice(path.insertionOp, baseShape);
    if (!insertedShape.has_value()) {
        return std::nullopt;
    }
    if (path.shapeTransforms.empty()) {
        return insertedShape;
    }

    TileInfo outputTile(insertedShape.value());
    for (auto shapeTransform : path.shapeTransforms | reversed) {
        const auto inputTiling = shapeTransform.backInferTileInfo(outputTile, _log.nest());
        if (inputTiling.tiles.size() != 1) {
            return std::nullopt;
        }
        outputTile = inputTiling.tiles.front();
    }
    return outputTile.shape;
}

std::optional<Shape> IterationShapeResolver::resolvePad(mlir::tensor::PadOp padOp, const Shape& baseShape, int depth) {
    auto srcShape = getValueShape(padOp.getSource(), depth + 1);
    if (!srcShape.has_value() || srcShape->size() != baseShape.size()) {
        return std::nullopt;
    }
    auto lowPads = padOp.getMixedLowPad();
    auto highPads = padOp.getMixedHighPad();
    if (lowPads.size() != baseShape.size() || highPads.size() != baseShape.size()) {
        return std::nullopt;
    }
    Shape result = baseShape;
    for (size_t i = 0; i < baseShape.size(); ++i) {
        auto low = eval(lowPads[i]);
        auto high = eval(highPads[i]);
        if (!low.has_value() || !high.has_value()) {
            return std::nullopt;
        }
        result[Dim(i)] = (*srcShape)[Dim(i)] + *low + *high;
    }
    return result;
}

/**
 * VPU.Expand widens its input by the static `pads_begin` / `pads_end` attributes.
 * Vertical Fusion can sink the Expand into the loop body, where it sits between the
 * input slice and the first NCE op of the chain. Without modelling it the back-track
 * from any NCE op of such a chain stops here and the whole chain falls back to the
 * bounded shape.
 */
std::optional<Shape> IterationShapeResolver::resolveExpand(VPU::ExpandOp expandOp, const Shape& baseShape, int depth) {
    auto srcShape = getValueShape(expandOp.getInput(), depth + 1);
    if (!srcShape.has_value() || srcShape->size() != baseShape.size()) {
        return std::nullopt;
    }
    const auto padsBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(expandOp.getPadsEnd());
    if (padsBegin.size() != baseShape.size() || padsEnd.size() != baseShape.size()) {
        return std::nullopt;
    }
    Shape result = baseShape;
    for (size_t i = 0; i < baseShape.size(); ++i) {
        result[Dim(i)] = (*srcShape)[Dim(i)] + padsBegin[i] + padsEnd[i];
    }
    return result;
}

/**
 * VPU.Slice carries its result extent in the static `static_sizes` attribute. An entry of
 * kDynamic means the slice keeps the whole source extent along that dimension - the same
 * rule VPU::SliceOp::reifyResultShapes encodes when it emits tensor.dim on the input - so
 * those dimensions are resolved from the source shape.
 *
 * Vertical Fusion sinks such a Slice into the loop body between two NCE ops (e.g. a
 * channel slice feeding NCE.CompressConvolution). Without modelling it the back-track from
 * the consumer stops here and the whole chain falls back to the bounded shape.
 */
std::optional<Shape> IterationShapeResolver::resolveVpuSlice(VPU::SliceOp sliceOp, const Shape& baseShape, int depth) {
    const auto staticSizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
    if (staticSizes.size() != baseShape.size()) {
        return std::nullopt;
    }
    std::optional<Shape> srcShape;
    Shape result = baseShape;
    for (size_t i = 0; i < baseShape.size(); ++i) {
        if (staticSizes[i] != mlir::ShapedType::kDynamic) {
            result[Dim(i)] = staticSizes[i];
            continue;
        }
        if (!srcShape.has_value()) {
            srcShape = getValueShape(sliceOp.getInput(), depth + 1);
            if (!srcShape.has_value() || srcShape->size() != baseShape.size()) {
                return std::nullopt;
            }
        }
        result[Dim(i)] = (*srcShape)[Dim(i)];
    }
    return result;
}

/**
 * Forward-infer the op's output shape from its resolved activation input.
 *
 * The inference itself lives on TilingBuilderOpInterface::fwdInferOutputTile - the forward
 * counterpart of backInferTileInfo used by resolveInsertionPath - so any op that models its own
 * forward mapping (NCE ops and Eltwise-trait ops out of the box) is handled here without this
 * resolver knowing the formula. Ops that do not model it return failure and the caller falls back
 * to the bounded shape.
 */
std::optional<Shape> IterationShapeResolver::inferOutputFromInput(VPU::NCEOpInterface nceOp, const Shape& baseShape,
                                                                  int depth) {
    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(nceOp.getOperation());
    if (tilingBuilder == nullptr) {
        return std::nullopt;
    }
    auto inShape = getValueShape(nceOp->getOperand(0), depth + 1);
    if (!inShape.has_value()) {
        return std::nullopt;
    }

    const SmallVector<vpux::TileInfo> inputTiles{vpux::TileInfo(ShapeRef(inShape.value()))};
    const auto outputTile = tilingBuilder.fwdInferOutputTile(inputTiles, _log.nest());
    if (mlir::failed(outputTile) || outputTile->shape.size() != baseShape.size()) {
        return std::nullopt;
    }
    return outputTile->shape;
}

}  // namespace vpux::VPU
