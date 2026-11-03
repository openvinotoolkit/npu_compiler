//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"

#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/core/IR/memref_attr.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/MemRef/Utils/MemRefUtils.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>

using namespace vpux;

namespace {

// Physical memory order implied by the strides: dims by descending stride, ties by descending bound.
DimsOrder inferMemoryOrderFromStrides(vpux::StridesRef bitStrides, vpux::BoundsRef bounds) {
    SmallVector<Dim> perm;
    for (size_t i = 0; i < bitStrides.size(); ++i) {
        perm.push_back(Dim(i));
    }
    llvm::stable_sort(perm, [&](Dim lhs, Dim rhs) {
        if (bitStrides[lhs].count() != bitStrides[rhs].count()) {
            return bitStrides[lhs].count() > bitStrides[rhs].count();
        }
        return bounds[lhs] > bounds[rhs];
    });
    return DimsOrder::fromPermutation(ArrayRef(perm));
}

// True if (order, strides) is non-overlapping w.r.t. `bounds`, i.e. builds a verifyLayout-valid MemRefAttr.
// getMemRefType() skips verifyLayout, so gate on this before committing a recovered order.
bool stridesValidForOrder(DimsOrder order, vpux::StridesRef bitStrides, vpux::BoundsRef bounds, Bit elemSize) {
    const auto memStrides = order.toMemoryOrder(bitStrides);
    const auto memShape = order.toMemoryOrder(ShapeRef(bounds.raw()));
    StrideReqs reqs;
    return reqs.checkStrides(memStrides, elemSize, memShape);
}

// Record `offset` on a vpux::MemRefAttr. Explicit strides are mandatory alongside an offset
// (getAffineMap() needs them); offset == 0 records nothing and returns `memref` unchanged.
mlir::MemRefType withRecordedOffset(mlir::MemRefType memref, const DimsOrder& order, vpux::StridesRef bitStrides,
                                    vpux::BoundsRef bounds, int64_t offset) {
    if (offset == 0) {
        return memref;
    }
    auto* ctx = memref.getContext();
    const auto elemSize = getElemTypeSize(memref.getElementType());
    const auto elemStrides = to_small_vector(bitStrides.raw() | transformed([&](Bit s) -> int64_t {
                                                 return mlir::ShapedType::isDynamic(s.count())
                                                                ? mlir::ShapedType::kDynamic
                                                                : s.count() / elemSize.count();
                                             }));
    auto layout = vpux::MemRefAttr::get(mlir::AffineMapAttr::get(order.toAffineMap(ctx)),
                                        getIntArrayAttr(ctx, elemStrides), nullptr, bounds, {}, ctx);
    return mlir::MemRefType::get(memref.getShape(), memref.getElementType(), layout.addOffset(offset),
                                 memref.getMemorySpace());
}

// Lower an inferred subview type to vpux::MemRefAttr form, attaching `bounds` as the slice's upper bounds.
mlir::MemRefType lowerSubViewResultType(mlir::MemRefType inferredType, mlir::MemRefType parentMemref,
                                        vpux::BoundsRef bounds) {
    // No bounds: keep the upstream layout for a plain-host parent (SW kernels must see no vpux::MemRefAttr),
    // a dynamic slice, or a contiguous row-major slice. Only a VPUX parent's non-contiguous static slice is
    // materialized, validated against its own shape.
    // E#236293 Investigate uniform memory-layout representation after tensor-slice bufferization
    vpux::BoundsRef validationShape = bounds;
    if (bounds.empty()) {
        if (!mlir::isa<vpux::MemRefAttr>(parentMemref.getLayout()) || !inferredType.hasStaticShape() ||
            mlir::memref::isStaticShapeAndContiguousRowMajor(inferredType)) {
            return inferredType;
        }
        validationShape = vpux::BoundsRef(inferredType.getShape());
    }

    // Already a vpux::MemRefAttr (bounds/offset carried): nothing to convert.
    if (mlir::isa<vpux::MemRefAttr>(inferredType.getLayout())) {
        return inferredType;
    }

    // Rank-reducing subviews are not supported (parent-rank order/strides can't map onto a reduced result).
    // Rank-preserving slices are all the tiled flows emit, so bail to the upstream type.
    if (inferredType.getRank() != parentMemref.getRank()) {
        return inferredType;
    }

    auto stridedLayout = mlir::dyn_cast<mlir::StridedLayoutAttr>(inferredType.getLayout());
    VPUX_THROW_WHEN(stridedLayout == nullptr,
                    "lowerSubViewResultType expects vpux::MemRefAttr or mlir::StridedLayoutAttr, got '{0}'",
                    inferredType);

    // Preserve the inferred offset (kDynamic for a runtime/loop-IV offset); dropping it to 0 makes the
    // subview verifier reject the op on an offset mismatch.
    const auto offset = stridedLayout.getOffset();

    const auto elemSize = getElemTypeSize(inferredType.getElementType());

    // Bit strides from the layout's explicit strides, NOT getStrides(), which recomputes compact strides for
    // a StridedLayoutAttr and would drop a non-compact/permuted parent's real layout.
    auto bitStrides = to_small_vector(stridedLayout.getStrides() | transformed([&](int64_t stride) {
                                          if (stride == mlir::ShapedType::kDynamic) {
                                              return Bit(mlir::ShapedType::kDynamic);
                                          }
                                          return Bit(stride * elemSize.count());
                                      }));

    // A dynamic inferred stride means a fully-dynamic parent with no vpux bounds to recover: keep the
    // upstream type. Static-stride strided parents fall through and get their order recovered below.
    if (llvm::any_of(bitStrides, [](Bit s) {
            return mlir::ShapedType::isDynamic(s.count());
        })) {
        return inferredType;
    }

    // A vpux::MemRefAttr parent records its order; any other parent's order is inferred from the strides
    // (identity order on non-monotonic strides would fail verifyLayout).
    DimsOrder order;
    if (auto parentMemRefAttr = mlir::dyn_cast<vpux::MemRefAttr>(parentMemref.getLayout())) {
        order = DimsOrder::fromAffineMap(parentMemRefAttr.order().getValue());
    } else {
        order = inferMemoryOrderFromStrides(StridesRef(bitStrides), validationShape);
    }

    // Commit only if the recovered order yields a verifyLayout-valid MemRefAttr; otherwise keep the upstream
    // type. Validate against validationShape (bounds for a dynamic slice, the static shape otherwise).
    if (!stridesValidForOrder(order, StridesRef(bitStrides), validationShape, elemSize)) {
        return inferredType;
    }

    // Match the parent memory space; the subview verifier rejects a mismatch.
    const auto memSpace = mlir::cast<vpux::NDTypeInterface>(parentMemref).getMemSpace();
    const auto resultType =
            vpux::getMemRefType(ShapeRef(inferredType.getShape()), inferredType.getElementType(), order, memSpace,
                                StridesRef(bitStrides), bounds, nullptr, nullptr, nullptr, nullptr);
    // Re-attach the offset with explicit strides (getMemRefType may have elided them as compact).
    return withRecordedOffset(resultType, order, StridesRef(bitStrides), bounds, offset);
}

// A MemRefAttr requires each static axis' bound to equal its extent.
SmallVector<int64_t> reconcileBoundsToStaticDims(vpux::BoundsRef inheritedBounds, mlir::ArrayRef<int64_t> shape) {
    SmallVector<int64_t> reconciled(inheritedBounds.raw().begin(), inheritedBounds.raw().end());
    for (int64_t idx = 0; idx < static_cast<int64_t>(shape.size()) && idx < static_cast<int64_t>(reconciled.size());
         ++idx) {
        const int64_t dim = shape[idx];
        if (!mlir::ShapedType::isDynamic(dim)) {
            reconciled[idx] = dim;
        }
    }
    return reconciled;
}

// Bounds for the sliced value: its own, else the peer's. Own bounds can be stale on a
// now-static axis, so reconcile them too.
SmallVector<int64_t> inheritSliceBounds(mlir::RankedTensorType sliceType, mlir::Type peerType) {
    if (sliceType.hasStaticShape()) {
        return {};
    }
    auto bounds = vpux::getBounds(sliceType);
    if (bounds.empty()) {
        bounds = vpux::getBounds(peerType);
    }
    if (bounds.empty()) {
        return {};
    }
    return reconcileBoundsToStaticDims(bounds, sliceType.getShape());
}

// ---- Upstream-derived bufferization models (above: VPUX helpers) ----

// Mirrors upstream ExtractSliceOpInterface; only getBufferType() differs, to carry VPUX bounds/offset.
struct TensorExtractSliceBufferizeModel final :
        public mlir::bufferization::BufferizableOpInterface::ExternalModel<TensorExtractSliceBufferizeModel,
                                                                           mlir::tensor::ExtractSliceOp> {
    bool bufferizesToMemoryRead(mlir::Operation*, mlir::OpOperand&, const mlir::bufferization::AnalysisState&) const {
        return false;
    }

    bool bufferizesToMemoryWrite(mlir::Operation*, mlir::OpOperand&, const mlir::bufferization::AnalysisState&) const {
        return false;
    }

    mlir::bufferization::AliasingValueList getAliasingValues(mlir::Operation* op, mlir::OpOperand&,
                                                             const mlir::bufferization::AnalysisState&) const {
        return {{op->getOpResult(0), mlir::bufferization::BufferRelation::Unknown}};
    }

    mlir::LogicalResult bufferize(mlir::Operation* op, mlir::RewriterBase& rewriter,
                                  const mlir::bufferization::BufferizationOptions& options,
                                  mlir::bufferization::BufferizationState& state) const {
        auto extractSliceOp = mlir::cast<mlir::tensor::ExtractSliceOp>(op);
        auto mixedOffsets = extractSliceOp.getMixedOffsets();
        auto mixedSizes = extractSliceOp.getMixedSizes();
        auto mixedStrides = extractSliceOp.getMixedStrides();

        auto srcMemref = mlir::bufferization::getBuffer(rewriter, extractSliceOp.getSource(), options, state);
        if (mlir::failed(srcMemref)) {
            return mlir::failure();
        }

        auto resultMemrefType = mlir::bufferization::getBufferType(extractSliceOp.getResult(), options, state);
        if (mlir::failed(resultMemrefType)) {
            return mlir::failure();
        }

        auto subView = mlir::memref::SubViewOp::create(rewriter, extractSliceOp.getLoc(),
                                                       mlir::cast<mlir::MemRefType>(*resultMemrefType), *srcMemref,
                                                       mixedOffsets, mixedSizes, mixedStrides);

        mlir::bufferization::replaceOpWithBufferizedValues(rewriter, op, subView.getResult());
        return mlir::success();
    }

    mlir::FailureOr<mlir::bufferization::BufferLikeType> getBufferType(
            mlir::Operation* op, mlir::Value value, const mlir::bufferization::BufferizationOptions& options,
            const mlir::bufferization::BufferizationState& state,
            llvm::SmallVector<mlir::Value>& invocationStack) const {
        auto extractSliceOp = mlir::cast<mlir::tensor::ExtractSliceOp>(op);
        VPUX_THROW_UNLESS(value == extractSliceOp.getResult(), "Invalid value for tensor.extract_slice");

        auto srcMemrefType =
                mlir::bufferization::getBufferType(extractSliceOp.getSource(), options, state, invocationStack);
        if (mlir::failed(srcMemrefType)) {
            return mlir::failure();
        }

        auto mixedOffsets = extractSliceOp.getMixedOffsets();
        auto mixedSizes = extractSliceOp.getMixedSizes();
        auto mixedStrides = extractSliceOp.getMixedStrides();

        auto inferredType = mlir::memref::SubViewOp::inferRankReducedResultType(
                extractSliceOp.getType().getShape(), mlir::cast<mlir::MemRefType>(*srcMemrefType), mixedOffsets,
                mixedSizes, mixedStrides);

        const SmallVector<int64_t> boundsStorage =
                inheritSliceBounds(extractSliceOp.getType(), extractSliceOp.getSource().getType());
        const auto bounds = vpux::BoundsRef(boundsStorage);

        auto resultType = lowerSubViewResultType(inferredType, mlir::cast<mlir::MemRefType>(*srcMemrefType), bounds);
        return mlir::cast<mlir::bufferization::BufferLikeType>(resultType);
    }
};

// Mirrors upstream InsertSliceOpInterface; bufferize() carries VPUX bounds/offset.
//
// This copy-free pipeline relies on the tiled flows emitting conflict-free (single-writer) accumulators.
struct TensorInsertSliceBufferizeModel final :
        public mlir::bufferization::DstBufferizableOpInterfaceExternalModel<TensorInsertSliceBufferizeModel,
                                                                            mlir::tensor::InsertSliceOp> {
    // Matches upstream insertSliceOpRequiresRead (the DstBufferizable base otherwise reads every operand).
    bool bufferizesToMemoryRead(mlir::Operation* op, mlir::OpOperand& opOperand,
                                const mlir::bufferization::AnalysisState&) const {
        auto insertSliceOp = mlir::cast<mlir::tensor::InsertSliceOp>(op);
        if (opOperand == insertSliceOp.getSourceMutable()) {
            return true;
        }
        const bool allOffsetsZero = llvm::all_of(insertSliceOp.getMixedOffsets(), mlir::isZeroInteger);
        const auto destType = insertSliceOp.getDestType();
        const bool sizesMatchDestSizes = mlir::areConstantIntValues(insertSliceOp.getMixedSizes(), destType.getShape());
        const bool allStridesOne = mlir::areAllConstantIntValue(insertSliceOp.getMixedStrides(), 1);
        return !(allOffsetsZero && sizesMatchDestSizes && allStridesOne);
    }

    mlir::LogicalResult bufferize(mlir::Operation* op, mlir::RewriterBase& rewriter,
                                  const mlir::bufferization::BufferizationOptions& options,
                                  mlir::bufferization::BufferizationState& state) const {
        auto insertSliceOp = mlir::cast<mlir::tensor::InsertSliceOp>(op);

        auto mixedOffsets = insertSliceOp.getMixedOffsets();
        auto mixedSizes = insertSliceOp.getMixedSizes();
        auto mixedStrides = insertSliceOp.getMixedStrides();

        auto dstMemref = mlir::bufferization::getBuffer(rewriter, insertSliceOp.getDest(), options, state);
        if (mlir::failed(dstMemref)) {
            return mlir::failure();
        }

        auto dstMemrefType = mlir::cast<mlir::MemRefType>((*dstMemref).getType());
        auto subviewMemRefType = mlir::memref::SubViewOp::inferRankReducedResultType(
                insertSliceOp.getSourceType().getShape(), dstMemrefType, mixedOffsets, mixedSizes, mixedStrides);

        const SmallVector<int64_t> boundsStorage =
                inheritSliceBounds(insertSliceOp.getSourceType(), insertSliceOp.getResult().getType());
        const auto bounds = vpux::BoundsRef(boundsStorage);

        auto loweredSubviewType = lowerSubViewResultType(subviewMemRefType, dstMemrefType, bounds);

        auto subView = mlir::memref::SubViewOp::create(rewriter, insertSliceOp.getLoc(), loweredSubviewType, *dstMemref,
                                                       mixedOffsets, mixedSizes, mixedStrides);

        auto srcMemref = mlir::bufferization::getBuffer(rewriter, insertSliceOp.getSource(), options, state);
        if (mlir::failed(srcMemref)) {
            return mlir::failure();
        }

        if (mlir::failed(options.memCpyFn(rewriter, insertSliceOp.getLoc(), *srcMemref, subView.getResult()))) {
            return mlir::failure();
        }

        mlir::bufferization::replaceOpWithBufferizedValues(rewriter, op, *dstMemref);
        return mlir::success();
    }

    mlir::FailureOr<mlir::bufferization::BufferLikeType> getBufferType(
            mlir::Operation* op, mlir::Value value, const mlir::bufferization::BufferizationOptions& options,
            const mlir::bufferization::BufferizationState& state,
            llvm::SmallVector<mlir::Value>& invocationStack) const {
        auto insertSliceOp = mlir::cast<mlir::tensor::InsertSliceOp>(op);
        VPUX_THROW_UNLESS(value == insertSliceOp.getResult(), "Invalid value for tensor.insert_slice");

        auto dstBufferType =
                mlir::bufferization::getBufferType(insertSliceOp.getDest(), options, state, invocationStack);
        if (mlir::failed(dstBufferType)) {
            return mlir::failure();
        }

        auto memRefType = mlir::dyn_cast<mlir::MemRefType>(*dstBufferType);
        if (memRefType == nullptr) {
            return *dstBufferType;
        }

        // The result aliases the destination buffer: return its type unchanged.
        return mlir::cast<mlir::bufferization::BufferLikeType>(memRefType);
    }
};

// Rebuild `srcMemref`'s layout with `dstBounds` (dropped when the shape is fully static), preserving the
// source order/strides: a tensor.cast never changes layout.
mlir::MemRefType relayoutMemRefWithBounds(mlir::MemRefType srcMemref, mlir::ArrayRef<int64_t> dstShape,
                                          vpux::BoundsRef dstBounds) {
    const auto srcNd = mlir::cast<vpux::NDTypeInterface>(srcMemref);
    const auto memSpace = srcNd.getMemSpace();
    const auto elemType = srcMemref.getElementType();

    // Reuse the source's physical strides. For a StridedLayoutAttr read them explicitly: getStrides()
    // returns compact identity strides for it and would discard a permuted/offset host layout. It is
    // faithful for vpux::MemRefAttr and bare-identity sources, so keep it there.
    SmallVector<Bit> srcBitStrides;
    if (auto stridedLayout = mlir::dyn_cast<mlir::StridedLayoutAttr>(srcMemref.getLayout())) {
        const auto elemSize = getElemTypeSize(elemType);
        srcBitStrides = to_small_vector(stridedLayout.getStrides() | transformed([&](int64_t stride) {
                                            return mlir::ShapedType::isDynamic(stride) ? Bit(mlir::ShapedType::kDynamic)
                                                                                       : Bit(stride * elemSize.count());
                                        }));
    } else if (!mlir::isa<vpux::MemRefAttr>(srcMemref.getLayout()) && !srcMemref.hasStaticShape()) {
        // A bare-identity dynamic source has no explicit strides: getStrides() -> calcStrides() asserts on a
        // dynamic shape. A cast can't re-stride anyway, so keep the source layout verbatim without bounds.
        // Must intercept before getStrides(); vpux::MemRefAttr sources read explicit strides and are excluded.
        return mlir::MemRefType::get(dstShape, elemType, srcMemref.getLayout(), srcMemref.getMemorySpace());
    } else {
        const auto strides = srcNd.getStrides();
        srcBitStrides.assign(strides.raw().begin(), strides.raw().end());
    }
    const auto srcStrides = StridesRef(srcBitStrides);

    // Recover the physical order: getDimsOrder() reports identity for a plain source, which is wrong for a
    // non-identity StridedLayoutAttr, so infer it from the strides in that case.
    DimsOrder order;
    if (mlir::isa<vpux::MemRefAttr>(srcMemref.getLayout()) || dstBounds.empty() || srcStrides.empty()) {
        order = srcNd.getDimsOrder();
    } else {
        // Dynamic strides can't be validated or bounded (checkStrides asserts static) and a cast can't
        // re-stride, so keep the source layout verbatim without bounds.
        if (llvm::any_of(srcStrides.raw(), [](Bit stride) {
                return mlir::ShapedType::isDynamic(stride.count());
            })) {
            return mlir::MemRefType::get(dstShape, elemType, srcMemref.getLayout(), srcMemref.getMemorySpace());
        }
        order = inferMemoryOrderFromStrides(srcStrides, dstBounds);
        // If the recovered order can't form a valid layout, keep the source layout verbatim (no bounds).
        if (!stridesValidForOrder(order, srcStrides, dstBounds, getElemTypeSize(elemType))) {
            return mlir::MemRefType::get(dstShape, elemType, srcMemref.getLayout(), srcMemref.getMemorySpace());
        }
    }

    auto resultType = vpux::getMemRefType(ShapeRef(dstShape), elemType, order, memSpace, srcStrides, dstBounds, nullptr,
                                          nullptr, nullptr, nullptr);

    // A cast aliases the same buffer, so keep the source offset (getMemRefType leaves it 0). Re-attach with
    // explicit strides, as getAffineMap() requires.
    int64_t srcOffset = 0;
    if (auto srcLayout = mlir::dyn_cast<vpux::MemRefAttr>(srcMemref.getLayout())) {
        if (auto offsetAttr = srcLayout.offset()) {
            srcOffset = offsetAttr.getInt();
        }
    } else if (auto stridedLayout = mlir::dyn_cast<mlir::StridedLayoutAttr>(srcMemref.getLayout())) {
        // A StridedLayoutAttr source carries its offset in the layout; forward it verbatim.
        srcOffset = stridedLayout.getOffset();
    }
    return withRecordedOffset(resultType, order, srcStrides, dstBounds, srcOffset);
}

// Implements upstream CastOpInterface for all three getBufferType cases. Cases 1 & 2 (unranked) mirror
// upstream via getMemRefTypeWithFullyDynamicLayout(). Case 3 (ranked) reuses the source layout when the
// source has no vpux::MemRefAttr and the destination needs no bounds (3a); otherwise it threads bounds
// through relayoutMemRefWithBounds() (3b).
struct TensorCastBufferizeModel final :
        public mlir::bufferization::BufferizableOpInterface::ExternalModel<TensorCastBufferizeModel,
                                                                           mlir::tensor::CastOp> {
    bool bufferizesToMemoryRead(mlir::Operation*, mlir::OpOperand&, const mlir::bufferization::AnalysisState&) const {
        return false;
    }

    bool bufferizesToMemoryWrite(mlir::Operation*, mlir::OpOperand&, const mlir::bufferization::AnalysisState&) const {
        return false;
    }

    mlir::bufferization::AliasingValueList getAliasingValues(mlir::Operation* op, mlir::OpOperand&,
                                                             const mlir::bufferization::AnalysisState&) const {
        return {{op->getOpResult(0), mlir::bufferization::BufferRelation::Equivalent}};
    }

    mlir::FailureOr<mlir::bufferization::BufferLikeType> getBufferType(
            mlir::Operation* op, mlir::Value value, const mlir::bufferization::BufferizationOptions& options,
            const mlir::bufferization::BufferizationState& state,
            llvm::SmallVector<mlir::Value>& invocationStack) const {
        auto castOp = mlir::cast<mlir::tensor::CastOp>(op);
        VPUX_THROW_UNLESS(value == castOp.getResult(), "Invalid value for tensor.cast");

        auto maybeSrcBufferType =
                mlir::bufferization::getBufferType(castOp.getSource(), options, state, invocationStack);
        if (mlir::failed(maybeSrcBufferType)) {
            return mlir::failure();
        }
        // Cases 1 & 2: either side unranked -> no layout/bounds to preserve, fall back to a fully-dynamic-
        // layout memref as upstream does. Gated on the tensor type (like upstream) so a ranked cast whose
        // source lowers to a non-memref buffer still reaches the ranked handling below.
        if (mlir::isa<mlir::UnrankedTensorType>(castOp.getSource().getType()) ||
            mlir::isa<mlir::UnrankedTensorType>(castOp.getType())) {
            const auto memorySpace = mlir::cast<mlir::BaseMemRefType>(*maybeSrcBufferType).getMemorySpace();
            auto resultType = mlir::bufferization::getMemRefTypeWithFullyDynamicLayout(castOp.getType(), memorySpace);
            return mlir::cast<mlir::bufferization::BufferLikeType>(resultType);
        }

        auto srcMemref = mlir::dyn_cast<mlir::MemRefType>(*maybeSrcBufferType);
        if (srcMemref == nullptr) {
            // Ranked source lowering to a non-memref (bounded/distributed) buffer: leave alone.
            return *maybeSrcBufferType;
        }

        auto dstTensor = mlir::cast<mlir::RankedTensorType>(castOp.getType());

        // Case 3a: reuse the source layout on the refined shape, as upstream does. A dynamic + bounded dst is
        // the exception -- reusing drops the bounds (no alloc size) -- so it falls through to Case 3b.
        if (!mlir::isa<vpux::MemRefAttr>(srcMemref.getLayout())) {
            const auto castDstBounds = vpux::getBounds(dstTensor);
            const bool dstNeedsBounds = !dstTensor.hasStaticShape() && !castDstBounds.empty();
            if (!dstNeedsBounds) {
                auto resultType = mlir::MemRefType::get(dstTensor.getShape(), srcMemref.getElementType(),
                                                        srcMemref.getLayout(), srcMemref.getMemorySpace());
                return mlir::cast<mlir::bufferization::BufferLikeType>(resultType);
            }
            // dst dynamic + bounded: fall through to Case 3b to keep the bounds.
        }

        // Fall back to the source bounds, then to a static source's shape as its exact bounds: a static ->
        // dynamic cast reaches here with empty dstBounds, and relayout would otherwise hand getMemRefType a
        // dynamic shape with explicit strides and no bounds, which it rejects.
        auto dstBounds = vpux::getBounds(dstTensor);
        if (dstBounds.empty()) {
            dstBounds = vpux::getBounds(srcMemref);
            if (dstBounds.empty() && srcMemref.hasStaticShape()) {
                dstBounds = vpux::BoundsRef(srcMemref.getShape());
            }
        }

        // Reconcile bounds per axis against the aliased source (the ground truth for static axes): a static
        // source axis pins the bound to its extent; otherwise clamp a bound exceeding a static result axis
        // down to it. Drop bounds entirely for a fully-static result.
        SmallVector<int64_t> clampedBounds;
        if (!dstBounds.empty()) {
            clampedBounds.assign(dstBounds.raw().begin(), dstBounds.raw().end());
            const auto srcShape = srcMemref.getShape();
            for (int64_t idx = 0; idx < dstTensor.getRank() && idx < static_cast<int64_t>(clampedBounds.size());
                 ++idx) {
                if (idx < static_cast<int64_t>(srcShape.size()) && !mlir::ShapedType::isDynamic(srcShape[idx])) {
                    clampedBounds[idx] = srcShape[idx];
                    continue;
                }
                const int64_t dim = dstTensor.getDimSize(idx);
                if (!mlir::ShapedType::isDynamic(dim) && clampedBounds[idx] > dim) {
                    clampedBounds[idx] = dim;
                }
            }
        }
        const bool dstIsStatic = llvm::none_of(dstTensor.getShape(), mlir::ShapedType::isDynamic);
        auto effectiveBounds = dstIsStatic ? vpux::BoundsRef{} : vpux::BoundsRef(clampedBounds);

        auto resultType = relayoutMemRefWithBounds(srcMemref, dstTensor.getShape(), effectiveBounds);
        return mlir::cast<mlir::bufferization::BufferLikeType>(resultType);
    }

    mlir::LogicalResult bufferize(mlir::Operation* op, mlir::RewriterBase& rewriter,
                                  const mlir::bufferization::BufferizationOptions& options,
                                  mlir::bufferization::BufferizationState& state) const {
        auto castOp = mlir::cast<mlir::tensor::CastOp>(op);

        auto srcBuffer = mlir::bufferization::getBuffer(rewriter, castOp.getSource(), options, state);
        if (mlir::failed(srcBuffer)) {
            return mlir::failure();
        }

        auto resultMemRefType = mlir::bufferization::getBufferType(castOp.getResult(), options, state);
        if (mlir::failed(resultMemRefType)) {
            return mlir::failure();
        }

        if (srcBuffer->getType() == *resultMemRefType) {
            mlir::bufferization::replaceOpWithBufferizedValues(rewriter, op, *srcBuffer);
            return mlir::success();
        }

        auto memRefCast = mlir::memref::CastOp::create(rewriter, castOp.getLoc(), *resultMemRefType, *srcBuffer);
        mlir::bufferization::replaceOpWithBufferizedValues(rewriter, op, memRefCast.getResult());
        return mlir::success();
    }
};

// tensor.empty intentionally has no custom model: upstream's EmptyOpInterface handles it, and bounds are
// threaded through vpux::getBufferType via options.unknownTypeConverterFn. A custom model would globally
// override empty allocation (dropping alignment, mishandling plain dynamic empties).

}  // namespace

void vpux::registerTensorSliceBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, mlir::tensor::TensorDialect*) {
        mlir::tensor::ExtractSliceOp::attachInterface<TensorExtractSliceBufferizeModel>(*ctx);
        mlir::tensor::InsertSliceOp::attachInterface<TensorInsertSliceBufferizeModel>(*ctx);
        mlir::tensor::CastOp::attachInterface<TensorCastBufferizeModel>(*ctx);
        // tensor.pad is intentionally not handled by these External Models.
    });
}
