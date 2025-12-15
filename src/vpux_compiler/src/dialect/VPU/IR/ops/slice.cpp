//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

using namespace vpux;

//
// build
//

void vpux::VPU::SliceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                               ShapeRef static_offsets, ShapeRef static_sizes) {
    build(builder, state, input, static_offsets.raw(), static_sizes.raw());
}

void vpux::VPU::SliceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                               ArrayRef<int64_t> static_offsets, ArrayRef<int64_t> static_sizes) {
    build(builder, state, input, getIntArrayAttr(builder.getContext(), static_offsets),
          getIntArrayAttr(builder.getContext(), static_sizes));
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::SliceOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                         mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                         mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                         mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::SliceOpAdaptor sliceOp(operands, attrs, prop);
    if (mlir::failed(sliceOp.verify(loc))) {
        return mlir::failure();
    }

    const auto origType = mlir::dyn_cast<vpux::NDTypeInterface>(sliceOp.getInput().getType());
    if (origType == nullptr) {
        return errorAt(loc, "VPU::SliceOp operand must have vpux::NDTypeInterface type");
    }

    const auto sliceShape = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
    const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());

    if (sliceShape.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Slice shape '{0}' doesn't match RankedTensor rank '{1}'", sliceShape, origType.getRank());
    }
    if (sliceOffsets.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Slice offsets '{0}' doesn't match RankedTensor rank '{1}'", sliceOffsets,
                       origType.getRank());
    }

    auto inferExplicitDistributedAttr = [&](VPU::DistributionInfoAttr origDistribution,
                                            ArrayRef<int64_t> inShape) -> VPU::DistributionInfoAttr {
        if (origDistribution.getMode().getValue() != VPU::DistributionMode::OVERLAPPED ||
            !VPU::isSegmentedOverlappedAxisSameAsSliceAxis(origDistribution.getNumTiles(), inShape, sliceShape)) {
            return VPU::getExplicitDistrAttrForSliceLikeOps(origDistribution, sliceShape, inShape, ctx);
        }

        // When clustering axis == slice axis, we cannot infer per cluster shape from op itself
        // and therefore this should be correctly computed in pass that creates the Slice Op
        auto memoryShapes = vpux::parseIntArrayOfArrayAttr<int64_t>(origDistribution.getMemoryShapes());

        for (size_t cluster = 0; cluster < memoryShapes.size(); cluster++) {
            for (size_t dim = 0; dim < inShape.size(); dim++) {
                // If this is the slice axis, the dim shape needs to be adjusted
                if (sliceShape[dim] != inShape[dim]) {
                    memoryShapes[cluster][dim] = sliceShape[dim];
                }
            }
        }
        const auto perClusterShapesAttr = vpux::getIntArrayOfArray(ctx, memoryShapes);
        const auto zeroOffsets =
                SmallVector<SmallVector<int64_t>>(memoryShapes.size(), SmallVector<int64_t>(inShape.size(), 0));
        const auto perClusterOffsetsAttr = vpux::getIntArrayOfArray(ctx, zeroOffsets);

        return VPU::DistributionInfoAttr::get(
                ctx, origDistribution.getMode(), origDistribution.getNumTiles(), origDistribution.getKernel(),
                origDistribution.getPads(), origDistribution.getStrides(), origDistribution.getNumClusters(),
                origDistribution.getAlignment(), origDistribution.getUniformDistributedSegments(), perClusterShapesAttr,
                perClusterOffsetsAttr, perClusterShapesAttr, perClusterOffsetsAttr,
                origDistribution.getEqualMemoryAndComputeView());
    };

    const auto distributedIn = mlir::dyn_cast<vpux::VPU::DistributedTypeInterface>(origType);
    VPU::DistributionInfoAttr possibleDistribution =
            distributedIn != nullptr && distributedIn.containsDistributedTypes()
                    ? mlir::cast<vpux::VPU::DistributedTensorType>(distributedIn.getDistributedTypes().front())
                              .getDistribution()
                    : nullptr;

    if (possibleDistribution != nullptr && VPU::isDistributedAttrWithExplicitShapesAndOffsets(possibleDistribution)) {
        if (auto sparseType = mlir::dyn_cast<vpux::VPU::SparseTensorType>(distributedIn)) {
            possibleDistribution = VPU::getExplicitDistrAttrForActualDataFromSparseType(sparseType);
        }

        auto newDistribution =
                VPU::updateSliceLikeOpsAlignment(ctx, origType.getShape(), ShapeRef(sliceShape), possibleDistribution);

        const auto sliceDistributedAttr = inferExplicitDistributedAttr(newDistribution, origType.getShape().raw());

        const auto newType = distributedIn.extractDenseTileForExplicitDistribution(
                ShapeRef(sliceOffsets), ShapeRef(sliceShape), sliceDistributedAttr);
        inferredTypes.emplace_back(newType);
    } else {
        const auto newType = origType.extractDenseTile(ShapeRef(sliceOffsets), ShapeRef(sliceShape));
        inferredTypes.emplace_back(newType);
    }

    return mlir::success();
}

mlir::LogicalResult VPU::SliceOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                    mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    SmallVector<mlir::OpFoldResult> shapes;
    auto loc = getOperation()->getLoc();
    auto outputShapedType = llvm::cast<mlir::ShapedType>(getOutput().getType());
    for (int64_t dim : llvm::seq<int64_t>(0, outputShapedType.getRank())) {
        if (!outputShapedType.isDynamicDim(dim)) {
            // Static dim: Return IntegerAttr.
            shapes.push_back(builder.getIndexAttr(outputShapedType.getDimSize(dim)));
        } else {
            // Dynamic dim: Return Value.
            mlir::OpFoldResult ofr = builder.createOrFold<mlir::tensor::DimOp>(loc, getInput(), dim);
            shapes.push_back(mlir::getValueOrCreateConstantIndexOp(builder, loc, ofr));
        }
    }
    reifiedReturnShapes.emplace_back(std::move(shapes));
    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult VPU::SliceOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    if (const auto origContent = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        const auto offset = Shape(parseIntArrayAttr<int64_t>(getStaticOffsets()));
        const auto shape = Shape(parseIntArrayAttr<int64_t>(getStaticSizes()));
        return static_cast<Const::ContentAttr>(origContent).transform().subview(offset, shape).get();
    }

    return nullptr;
}

//
// verify
//

mlir::LogicalResult vpux::VPU::SliceOp::verify() {
    const auto loc = getLoc();

    const auto inShape = getShape(getInput());
    const auto outShape = getShape(getOutput());
    const auto sliceOffsets = parseIntArrayAttr<int64_t>(getStaticOffsets());
    if (inShape.size() != outShape.size()) {
        return errorAt(loc, "Input shape '{0}' and output shape '{1}' must have the same rank", inShape, outShape);
    }
    if (inShape.size() != sliceOffsets.size()) {
        return errorAt(loc, "Input shape '{0}' and slice offsets '{1}' must have the same rank", inShape, sliceOffsets);
    }

    const auto sliceDims = IE::getDiffInOutSizeDims(inShape, outShape);

    for (auto idx : irange(inShape.size())) {
        auto dim = Dim(idx);
        auto isSliceDim = llvm::find(sliceDims, dim) != sliceDims.end();
        if (isSliceDim) {
            if (outShape[dim] + sliceOffsets[idx] > inShape[dim]) {
                return errorAt(loc,
                               "Slice offset '{0}' and output shape '{1}' exceed input shape '{2}' on dimension '{3}'",
                               sliceOffsets[idx], outShape[dim], inShape[dim], dim);
            }
        } else {
            if (inShape[dim] != outShape[dim]) {
                return errorAt(loc, "Input shape '{0}' and output shape '{1}' must match on non-sliced dimensions",
                               inShape, outShape);
            }
            if (sliceOffsets[idx] != 0) {
                return errorAt(loc, "Slice offset '{0}' for non-sliced dimension '{1}' must be zero", sliceOffsets[idx],
                               dim);
            }
        }
    }
    return mlir::success();
}

//
// ComposeSlice
//

namespace {

class ComposeSlice final : public mlir::OpRewritePattern<VPU::SliceOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::SliceOp origOp, mlir::PatternRewriter& rewriter) const final {
        auto producerSliceOp = origOp.getInput().getDefiningOp<VPU::SliceOp>();
        if (producerSliceOp == nullptr) {
            return mlir::failure();
        }

        auto finalOffsets = parseIntArrayAttr<int64_t>(producerSliceOp.getStaticOffsets());
        const auto secondOffsets = parseIntArrayAttr<int64_t>(origOp.getStaticOffsets());
        for (auto i : irange(finalOffsets.size())) {
            finalOffsets[i] += secondOffsets[i];
        }

        const auto finalOffsetsAttr = getIntArrayAttr(getContext(), finalOffsets);
        const auto finalShapeAttr = origOp.getStaticSizes();
        rewriter.replaceOpWithNewOp<VPU::SliceOp>(origOp, producerSliceOp.getInput(), finalOffsetsAttr, finalShapeAttr);

        return mlir::success();
    }
};

// Remove redundant pairs of Expand->Slice operations which negate each other's effects For example:
//
// Case 1. Only expand at the end
// [1, 16, 1, 1]
//   -> Expand {pads_begin = [0, 0, 0, 0], pads_end = [15, 0, 0, 0]} -> [16, 16, 1, 1]
//   -> Slice  {offsets =    [0, 0, 0, 0], sizes =    [1, 16, 1, 1]} -> [1, 16, 1, 1]
//
// Case 2. Expand on both sides
// [1, 16, 1, 1]
//   -> Expand {pads_begin = [4, 0, 0, 0], pads_end = [15, 0, 0, 0]} -> [20, 16, 1, 1]
//   -> Slice  {offsets =    [4, 0, 0, 0], sizes =    [1, 16, 1, 1]} -> [1, 16, 1, 1]
class RemoveRedundantExpandSlice final : public mlir::OpRewritePattern<VPU::SliceOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::SliceOp sliceOp, mlir::PatternRewriter& rewriter) const final {
        auto expandOp = sliceOp.getInput().getDefiningOp<VPU::ExpandOp>();
        if (expandOp == nullptr) {
            return mlir::failure();
        }

        const auto origInputShape = getShape(expandOp.getInput());
        const auto origOutputShape = getShape(sliceOp.getOutput());
        if (origInputShape != origOutputShape) {
            return mlir::failure();
        }

        const auto expandPadsBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin());
        const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
        for (size_t i = 0; i < origInputShape.size(); ++i) {
            if (sliceOffsets[i] - expandPadsBegin[i] != 0) {
                return mlir::failure();
            }
        }

        rewriter.replaceAllOpUsesWith(sliceOp, expandOp.getInput());

        return mlir::success();
    }
};

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::SliceOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<ComposeSlice>(ctx);
    results.add<RemoveRedundantExpandSlice>(ctx);
}
