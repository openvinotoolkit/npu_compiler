//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/back_infer_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/distributed_buffer_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// build
//

void VPUIP::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                             ShapeRef static_offsets, ShapeRef static_sizes) {
    build(builder, state, input, static_offsets.raw(), static_sizes.raw());
}

void VPUIP::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                             ArrayRef<int64_t> static_offsets, ArrayRef<int64_t> static_sizes) {
    build(builder, state, input, getIntArrayAttr(builder.getContext(), static_offsets),
          getIntArrayAttr(builder.getContext(), static_sizes));
}

void VPUIP::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                             mlir::ArrayAttr static_offsets, mlir::ArrayAttr static_sizes) {
    build(builder, state, input, static_offsets, static_sizes, nullptr, nullptr, nullptr);
}

void VPUIP::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                             mlir::ArrayAttr static_offsets, mlir::ArrayAttr static_sizes,
                             mlir::ArrayAttr static_strides) {
    build(builder, state, input, static_offsets, static_sizes, static_strides, nullptr, nullptr);
}

void VPUIP::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                             ShapeRef static_offsets, ShapeRef static_sizes, ShapeRef static_strides) {
    build(builder, state, input, static_offsets.raw(), static_sizes.raw(), static_strides.raw());
}

void VPUIP::SubViewOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                             ArrayRef<int64_t> static_offsets, ArrayRef<int64_t> static_sizes,
                             ArrayRef<int64_t> static_strides) {
    build(builder, state, input, getIntArrayAttr(builder.getContext(), static_offsets),
          getIntArrayAttr(builder.getContext(), static_sizes), getIntArrayAttr(builder.getContext(), static_strides),
          nullptr, nullptr);
}

//
// ViewLikeOpInterface
//

mlir::Value VPUIP::SubViewOp::getViewSource() {
    return getSource();
}

namespace {

bool isValidSubViewTile(vpux::NDTypeInterface inputType, ArrayRef<int64_t> staticOffsets, ArrayRef<int64_t> staticSizes,
                        ArrayRef<int64_t> staticStrides) {
    const auto rank = checked_cast<size_t>(inputType.getRank());
    if (staticOffsets.size() != rank || staticSizes.size() != rank || staticStrides.size() != rank) {
        return false;
    }

    const auto inputShape = inputType.getShape();
    for (auto idx : irange(rank)) {
        const auto offset = staticOffsets[idx];
        const auto size = staticSizes[idx];
        const auto stride = staticStrides[idx];
        if (offset < 0 || size <= 0 || stride <= 0) {
            return false;
        }

        const auto dimSize = inputShape[Dim(checked_cast<int32_t>(idx))];
        if (offset >= dimSize || (size - 1) > (dimSize - offset - 1) / stride) {
            return false;
        }
    }

    return true;
}

bool isValidTileElementType(mlir::Type elemType, ArrayRef<int64_t> staticOffsets, ArrayRef<int64_t> staticSizes) {
    const auto perAxisType = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(elemType);
    if (perAxisType == nullptr) {
        return true;
    }

    const auto quantizedAxis = perAxisType.getQuantizedDimension();
    if (quantizedAxis < 0 || checked_cast<size_t>(quantizedAxis) >= staticSizes.size()) {
        return false;
    }
    const auto numScales = checked_cast<int64_t>(perAxisType.getScales().size());
    return staticOffsets[quantizedAxis] + staticSizes[quantizedAxis] <= numScales;
}

bool isValidTileSparsityCompression(VPUIP::SparsityCompressionAttr sparsityCompression, ArrayRef<int64_t> staticOffsets,
                                    ArrayRef<int64_t> staticSizes) {
    if (sparsityCompression == nullptr) {
        return true;
    }
    const auto axisAttr = sparsityCompression.getAxis();
    if (axisAttr == nullptr) {
        return false;
    }

    const auto axis = checked_cast<size_t>(axisAttr.getInt());
    if (axis >= staticOffsets.size() || axis >= staticSizes.size()) {
        return false;
    }

    const auto numElems = sparsityCompression.getNumElems().getValues<int64_t>();
    return staticOffsets[axis] + staticSizes[axis] <= checked_cast<int64_t>(numElems.size());
}

bool isValidExplicitSliceDistribution(VPU::DistributionInfoAttr distribution) {
    const auto mode = distribution.getMode().getValue();
    return mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED ||
           VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) ||
           VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED);
}

SmallVector<int64_t> parseStaticStrides(std::optional<mlir::ArrayAttr> staticStridesAttr, int64_t rank) {
    return staticStridesAttr.has_value() ? parseIntArrayAttr<int64_t>(staticStridesAttr.value())
                                         : SmallVector<int64_t>(checked_cast<size_t>(rank), 1);
}

std::optional<mlir::Type> inferSubViewOutputType(mlir::Type inputType, ArrayRef<int64_t> staticOffsets,
                                                 ArrayRef<int64_t> staticSizes, ArrayRef<int64_t> staticStrides) {
    // Infer the result of slicing a candidate input type without requiring an
    // actual SubViewOp. Example: a planned tile offsets [0, 0, 16, 0],
    // sizes [1, 16, 16, 32] over a 1x16x32x32 input should produce the same
    // type that SubViewOp::inferReturnTypes would later create.
    auto newInputNDType = mlir::dyn_cast_or_null<vpux::NDTypeInterface>(inputType);
    if (newInputNDType == nullptr || !isValidSubViewTile(newInputNDType, staticOffsets, staticSizes, staticStrides)) {
        return std::nullopt;
    }
    if (!isValidTileElementType(newInputNDType.getElementType(), staticOffsets, staticSizes)) {
        return std::nullopt;
    }

    // SubView back inference may test distributed slice candidates that are not
    // legal. Keep that path quiet and return std::nullopt; real diagnostics are
    // still emitted by the verifier.
    mlir::ScopedDiagnosticHandler diagnosticGuard(inputType.getContext(), [](mlir::Diagnostic&) {
        return mlir::success();
    });
    if (auto distType = mlir::dyn_cast<VPUIP::DistributedBufferType>(newInputNDType)) {
        auto distribution = distType.getDistribution();
        if (distribution.getMode().getValue() == VPU::DistributionMode::OVERLAPPED &&
            VPU::isSegmentedOverlappedAxisSameAsSliceAxis(distribution.getNumTiles(), newInputNDType.getShape().raw(),
                                                          staticSizes)) {
            return std::nullopt;
        }
        if (!isValidTileSparsityCompression(distType.getSparsityCompression(), staticOffsets, staticSizes)) {
            return std::nullopt;
        }

        auto* ctx = inputType.getContext();
        auto newDistribution =
                VPU::updateSliceLikeOpsAlignment(ctx, newInputNDType.getShape(), ShapeRef(staticSizes), distribution);
        if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(newDistribution)) {
            // Explicit distributions must be sliced with explicit
            // per-cluster shapes/offsets. For example, slicing H from 32 to 16
            // has to update each cluster's memory view before extracting the
            // tile.
            if (!isValidExplicitSliceDistribution(newDistribution)) {
                return std::nullopt;
            }
            const auto sliceDistAttr = VPU::getExplicitDistrAttrForSliceLikeOps(newDistribution, staticSizes,
                                                                                newInputNDType.getShape().raw(), ctx);
            auto outType = distType.extractViewTileForExplicitDistribution(
                    ShapeRef(staticOffsets), ShapeRef(staticSizes), ShapeRef(staticStrides), sliceDistAttr);
            return mlir::cast<mlir::Type>(outType);
        }
        auto newBufferType = VPUIP::createDistributedBufferTypeOrNull(
                ctx, distType.getShape(), distType.getElementType(), distType.getLayout(), distType.getMemSpace(),
                newDistribution, distType.getSparsityCompression());
        if (!newBufferType.has_value()) {
            return std::nullopt;
        }
        auto outType = newBufferType.value().extractViewTile(ShapeRef(staticOffsets), ShapeRef(staticSizes),
                                                             ShapeRef(staticStrides));
        return mlir::cast<mlir::Type>(outType);
    }

    if (!isValidTileSparsityCompression(VPUIP::getSparsityCompressionAttr(inputType), staticOffsets, staticSizes)) {
        return std::nullopt;
    }
    auto outType =
            newInputNDType.extractViewTile(ShapeRef(staticOffsets), ShapeRef(staticSizes), ShapeRef(staticStrides));
    return mlir::cast<mlir::Type>(outType);
}

}  // namespace

std::optional<mlir::Type> VPUIP::inferSubViewOutputTypeFromTile(mlir::Type inputType, ArrayRef<int64_t> staticOffsets,
                                                                ArrayRef<int64_t> staticSizes) {
    auto inputNDType = mlir::dyn_cast_or_null<vpux::NDTypeInterface>(inputType);
    if (inputNDType == nullptr) {
        return std::nullopt;
    }

    SmallVector<int64_t> unitStrides(checked_cast<size_t>(inputNDType.getRank()), 1);
    return inferSubViewOutputType(inputType, staticOffsets, staticSizes, unitStrides);
}

std::optional<mlir::Type> VPUIP::inferSubViewOutputTypeFromTile(mlir::Type inputType, ArrayRef<int64_t> staticOffsets,
                                                                ArrayRef<int64_t> staticSizes,
                                                                ArrayRef<int64_t> staticStrides) {
    return inferSubViewOutputType(inputType, staticOffsets, staticSizes, staticStrides);
}

//
// BackInferViewTypeOpInterface
//

std::optional<mlir::Type> VPUIP::SubViewOp::inferOutputTypeFromInput(mlir::Type newInputType) {
    auto newInputNDType = mlir::dyn_cast<vpux::NDTypeInterface>(newInputType);
    if (newInputNDType == nullptr) {
        return std::nullopt;
    }

    const auto staticSizes = parseIntArrayAttr<int64_t>(getStaticSizes());
    const auto staticOffsets = parseIntArrayAttr<int64_t>(getStaticOffsets());
    const auto staticStrides = parseStaticStrides(getStaticStrides(), newInputNDType.getRank());
    return inferSubViewOutputType(newInputType, staticOffsets, staticSizes, staticStrides);
}

std::optional<mlir::Type> VPUIP::SubViewOp::inferInputTypeFromOutput(mlir::Type) {
    return std::nullopt;
}

//
// InferTypeOpInterface
//

mlir::LogicalResult VPUIP::SubViewOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                       mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                       mlir::OpaqueProperties props, mlir::RegionRange /*regions*/,
                                                       mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPUIP::SubViewOpAdaptor subViewOp(operands, attrs, props);
    if (mlir::failed(subViewOp.verify(loc))) {
        return mlir::failure();
    }

    const auto origType = mlir::cast<vpux::NDTypeInterface>(subViewOp.getSource().getType());

    const auto subViewShape = parseIntArrayAttr<int64_t>(subViewOp.getStaticSizes());
    const auto subViewOffsets = parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsets());
    const auto subViewStrides = parseStaticStrides(subViewOp.getStaticStrides(), origType.getRank());

    if (subViewShape.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Tile shape '{0}' doesn't match MemRef rank '{1}'", subViewShape, origType.getRank());
    }
    if (subViewOffsets.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Tile offsets '{0}' doesn't match MemRef rank '{1}'", subViewOffsets, origType.getRank());
    }
    if (subViewStrides.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Tile strides '{0}' doesn't match MemRef rank '{1}'", subViewStrides, origType.getRank());
    }

    const auto hasExplicitOutputShapes = subViewOp.getExplicitOutputShapes().has_value();
    const auto hasExplicitOutputOffsets = subViewOp.getExplicitOutputOffsets().has_value();

    auto inferExplicitDistributedAttr = [&](VPU::DistributionInfoAttr origDistribution,
                                            ArrayRef<int64_t> inShape) -> VPU::DistributionInfoAttr {
        auto mode = origDistribution.getMode().getValue();
        if (hasExplicitOutputShapes && hasExplicitOutputOffsets) {
            // When both explicit output shapes and offsets are provided (e.g. for Overlapped mode),
            // use them directly to construct the distribution attribute
            return VPU::DistributionInfoAttr::get(
                    ctx, origDistribution.getMode(), origDistribution.getNumTiles(), origDistribution.getKernel(),
                    origDistribution.getPads(), origDistribution.getStrides(), origDistribution.getNumClusters(),
                    origDistribution.getAlignment(), origDistribution.getUniformDistributedSegments(),
                    subViewOp.getExplicitOutputShapes().value(), subViewOp.getExplicitOutputOffsets().value(),
                    subViewOp.getExplicitOutputShapes().value(), subViewOp.getExplicitOutputOffsets().value(),
                    origDistribution.getEqualMemoryAndComputeView(), origDistribution.getMemoryNumTiles());
        }
        if (hasExplicitOutputShapes) {
            // Track #E125638
            // Other modes should be supported
            VPUX_THROW_UNLESS(mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED,
                              "Can not set explicit shapes with mode {0}", VPU::stringifyDistributionMode(mode));
            auto explicitOutputShapes = subViewOp.getExplicitOutputShapes().value();
            return VPU::getSegmentedExplicitDistrAttrForSliceLikeOps(origDistribution, subViewShape,
                                                                     explicitOutputShapes, ctx);
        }
        if (mode != VPU::DistributionMode::OVERLAPPED ||
            !VPU::isSegmentedOverlappedAxisSameAsSliceAxis(origDistribution.getNumTiles(), inShape, subViewShape)) {
            return VPU::getExplicitDistrAttrForSliceLikeOps(origDistribution, subViewShape, inShape, ctx);
        }

        // When clustering axis == slice axis, we cannot infer per cluster shape from op itself
        // and therefore this should be correctly computed in pass that creates the Subview Op
        auto memoryShapes = vpux::parseIntArrayOfArrayAttr<int64_t>(origDistribution.getMemoryShapes());

        for (size_t cluster = 0; cluster < memoryShapes.size(); cluster++) {
            for (size_t dim = 0; dim < inShape.size(); dim++) {
                // If this is the slice axis, the dim shape needs to be adjusted
                if (subViewShape[dim] != inShape[dim]) {
                    memoryShapes[cluster][dim] = subViewShape[dim];
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
                origDistribution.getEqualMemoryAndComputeView(), origDistribution.getMemoryNumTiles());
    };

    const auto distributedIn = mlir::dyn_cast<vpux::VPU::DistributedTypeInterface>(origType);
    VPU::DistributionInfoAttr possibleDistribution =
            distributedIn != nullptr && distributedIn.containsDistributedTypes()
                    ? mlir::cast<vpux::VPUIP::DistributedBufferType>(distributedIn.getDistributedTypes().front())
                              .getDistribution()
                    : nullptr;
    if (possibleDistribution != nullptr) {
        if (hasExplicitOutputShapes || hasExplicitOutputOffsets ||
            VPU::isDistributedAttrWithExplicitShapesAndOffsets(possibleDistribution)) {
            if (auto sparseType = mlir::dyn_cast<vpux::VPUIP::SparseBufferType>(distributedIn)) {
                possibleDistribution = VPU::getExplicitDistrAttrForActualDataFromSparseType(sparseType);
            }

            // update subview alignment
            auto newDistribution = VPU::updateSliceLikeOpsAlignment(ctx, origType.getShape(), ShapeRef(subViewShape),
                                                                    possibleDistribution);

            const auto subViewDistributedAttr =
                    inferExplicitDistributedAttr(newDistribution, origType.getShape().raw());

            const auto subViewType = distributedIn.extractViewTileForExplicitDistribution(
                    ShapeRef(subViewOffsets), ShapeRef(subViewShape), ShapeRef(subViewStrides), subViewDistributedAttr);
            inferredTypes.push_back(subViewType);
        } else {
            // todo: update alignment for non-explict sparseBuffer to enable 37XX unaligned shave tiling
            // ticket E#114487
            if (auto sparseType = mlir::dyn_cast<vpux::VPUIP::SparseBufferType>(distributedIn)) {
                const auto subViewType = sparseType.extractViewTile(ShapeRef(subViewOffsets), ShapeRef(subViewShape),
                                                                    ShapeRef(subViewStrides));
                inferredTypes.push_back(subViewType);
            } else {
                auto newDistribution = VPU::updateSliceLikeOpsAlignment(ctx, origType.getShape(),
                                                                        ShapeRef(subViewShape), possibleDistribution);

                const auto origBufferType =
                        mlir::cast<vpux::VPUIP::DistributedBufferType>(distributedIn.getDistributedTypes().front());
                auto newBufferType = VPUIP::DistributedBufferType::get(
                        ctx, origBufferType.getShape().raw(), origBufferType.getElementType(),
                        origBufferType.getLayout(), origBufferType.getMemSpace(), newDistribution,
                        origBufferType.getSparsityCompression());

                const auto subViewType = newBufferType.extractViewTile(ShapeRef(subViewOffsets), ShapeRef(subViewShape),
                                                                       ShapeRef(subViewStrides));
                inferredTypes.push_back(subViewType);
            }
        }
    } else {
        const auto subViewType =
                origType.extractViewTile(ShapeRef(subViewOffsets), ShapeRef(subViewShape), ShapeRef(subViewStrides));

        inferredTypes.push_back(subViewType);
    }

    return mlir::success();
}

// A sparsity map constant has the workload size flattened, so that its shape is OCx1x1xSIZE.
// Therefore, only subviews over the OC dimension are allowed.
// Additionally, the GetSparsityMap transformation is the last one in the list. When folding
// subviews into the constant, it will be introduced as a transformation before it, so its
// subview dimensions have to be adapted for the shape before flattening.
void adaptSparsityMapConstant(mlir::Value source, Shape& offset, Shape& shape) {
    auto constParentOp = source.getDefiningOp<Const::DeclareOp>();
    if (constParentOp == nullptr) {
        return;
    }
    const auto transformations = constParentOp.getContentAttr().getTransformations();
    if (transformations.empty()) {
        return;
    }

    auto getSparistyMapTransIt = std::find_if(transformations.rbegin(), transformations.rend(),
                                              [&](vpux::Const::TransformAttrInterface trans) {
                                                  return mlir::isa<vpux::Const::GetSparsityMapAttr>(trans);
                                              });
    if (getSparistyMapTransIt == transformations.rend()) {
        return;
    }

    auto posFromEnd = std::distance(transformations.rbegin(), getSparistyMapTransIt);

    const auto zeroWorkloadOffsets = std::all_of(offset.begin() + 1, offset.end(), [](const int64_t value) {
        return value == 0;
    });
    VPUX_THROW_UNLESS(zeroWorkloadOffsets, "Offsets with non-zero values for workloads are not supported. Got {0}",
                      offset);

    const auto sparsityMapShape = mlir::cast<vpux::NDTypeInterface>(constParentOp.getType()).getShape();
    const auto sparsityMapWorkloadShape = SmallVector<int64_t>(sparsityMapShape.begin() + 1, sparsityMapShape.end());
    const auto shapeWorkloadShape = SmallVector<int64_t>(shape.begin() + 1, shape.end());
    for (auto p : zip(sparsityMapWorkloadShape, shapeWorkloadShape)) {
        const auto sparsityMapDim = std::get<0>(p);
        const auto shapeDim = std::get<1>(p);
        VPUX_THROW_UNLESS(sparsityMapDim == shapeDim,
                          "Subview shape with different workload size is not supported: original dim {0}, new dim {1}",
                          sparsityMapDim, shapeDim);
    }

    auto inputType = mlir::cast<vpux::NDTypeInterface>(constParentOp.getContentAttr().getBaseContent().getType());
    for (auto idx : irange(transformations.size() - (1 + posFromEnd))) {
        inputType = transformations[idx].inferOutputType(inputType);
    }
    const auto inputShape = inputType.getShape().raw();
    VPUX_THROW_UNLESS(inputShape.size() == 4, "Expected a 4-dimensional type, got {0} dimensions", inputShape.size());
    const auto OC = shape.raw()[0];
    shape = Shape({OC, inputShape[1], inputShape[2], inputShape[3]});
}

//
// fold
//

mlir::OpFoldResult VPUIP::SubViewOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (getSource().getType() == getResult().getType()) {
        return getSource();
    }

    if (const auto origContent = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        auto offset = Shape(parseIntArrayAttr<int64_t>(getStaticOffsets()));
        auto shape = Shape(parseIntArrayAttr<int64_t>(getStaticSizes()));
        adaptSparsityMapConstant(getSource(), offset, shape);
        return static_cast<Const::ContentAttr>(origContent).transform().subview(offset, shape).get();
    }

    return nullptr;
}

//
// ComposeSubView
//

namespace {

class ComposeSubView final : public mlir::OpRewritePattern<VPUIP::SubViewOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPUIP::SubViewOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ComposeSubView::matchAndRewrite(VPUIP::SubViewOp origOp, mlir::PatternRewriter& rewriter) const {
    auto producerSubViewOp = origOp.getSource().getDefiningOp<VPUIP::SubViewOp>();
    if (producerSubViewOp == nullptr) {
        return mlir::failure();
    }

    if (origOp.getStaticStrides().has_value() || producerSubViewOp.getStaticStrides().has_value()) {
        return mlir::failure();
    }

    auto finalOffsets = parseIntArrayAttr<int64_t>(producerSubViewOp.getStaticOffsets());
    const auto secondOffsets = parseIntArrayAttr<int64_t>(origOp.getStaticOffsets());
    for (auto i : irange(finalOffsets.size())) {
        finalOffsets[i] += secondOffsets[i];
    }

    const auto finalOffsetsAttr = getIntArrayAttr(getContext(), finalOffsets);
    const auto finalShapeAttr = origOp.getStaticSizes();
    rewriter.replaceOpWithNewOp<VPUIP::SubViewOp>(origOp, producerSubViewOp.getSource(), finalOffsetsAttr,
                                                  finalShapeAttr);

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void VPUIP::SubViewOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<ComposeSubView>(ctx);
}

//
// verify
//

mlir::LogicalResult VPUIP::SubViewOp::verify() {
    const auto op = getOperation();
    const auto logCb = [op](const formatv_object_base& msg) {
        std::ignore = errorAt(op, "{0}", msg.str());
    };

    if (getExplicitOutputShapes().has_value() != getExplicitOutputOffsets().has_value()) {
        return errorAt(op, "explicit_output_shapes and explicit_output_offsets must be set together");
    }

    mlir::SmallVector<mlir::Type> inferredTypes;
    if (inferReturnTypes(getContext(), getLoc(), op->getOperands(), op->getAttrDictionary(), op->getPropertiesStorage(),
                         op->getRegions(), inferredTypes)
                .failed()) {
        logCb(formatv("Can't infer return types"));
        return mlir::failure();
    }
    const auto expectedStrides = mlir::cast<vpux::NDTypeInterface>(inferredTypes.front()).getStrides();
    const auto outputStrides = mlir::cast<vpux::NDTypeInterface>(getResult().getType()).getStrides();
    if (expectedStrides.size() != outputStrides.size()) {
        logCb(formatv("The output stride size != infered stride size"));
        return mlir::failure();
    }

    for (auto j : irange(expectedStrides.size())) {
        if (outputStrides[Dim(j)] != expectedStrides[Dim(j)]) {
            logCb(formatv("The output stride({0}) != infered stride({1})", outputStrides, expectedStrides));
            return mlir::failure();
        }
    }
    return mlir::success();
}
