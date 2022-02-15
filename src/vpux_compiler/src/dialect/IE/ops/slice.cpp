//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// build
//

void vpux::IE::SliceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                              ShapeRef static_offsets, ShapeRef static_sizes) {
    build(builder, state, input, static_offsets.raw(), static_sizes.raw());
}

void vpux::IE::SliceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                              ArrayRef<int64_t> static_offsets, ArrayRef<int64_t> static_sizes) {
    build(builder, state, input, getIntArrayAttr(builder.getContext(), static_offsets),
          getIntArrayAttr(builder.getContext(), static_sizes));
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::IE::SliceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::SliceOpAdaptor sliceOp(operands, attrs);
    if (mlir::failed(sliceOp.verify(loc))) {
        return mlir::failure();
    }

    const auto origType = sliceOp.source().getType().dyn_cast<vpux::NDTypeInterface>();
    if (origType == nullptr) {
        return errorAt(loc, "IE::SliceOp operand must have vpux::NDTypeInterface type");
    }

    const auto sliceShape = parseIntArrayAttr<int64_t>(sliceOp.static_sizes());
    const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.static_offsets());

    if (sliceShape.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Slice shape '{0}' doesn't match RankedTensor rank '{1}'", sliceShape, origType.getRank());
    }
    if (sliceOffsets.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Slice offsets '{0}' doesn't match RankedTensor rank '{1}'", sliceOffsets,
                       origType.getRank());
    }

    const auto newType = origType.extractDenseTile(ShapeRef(sliceOffsets), ShapeRef(sliceShape));
    const auto newTensorType = newType.cast<mlir::RankedTensorType>();
    inferredReturnShapes.emplace_back(newTensorType.getShape(), newTensorType.getElementType(),
                                      newTensorType.getEncoding());

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::SliceOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (source().getType() == result().getType()) {
        return source();
    }

    if (const auto origContent = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        const auto offset = Shape(parseIntArrayAttr<int64_t>(static_offsets()));
        const auto shape = Shape(parseIntArrayAttr<int64_t>(static_sizes()));
        return origContent.subview(offset, shape);
    }

    return nullptr;
}

//
// ComposeSlice
//

namespace {

class ComposeSlice final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(IE::SliceOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ComposeSlice::matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const {
    auto producerSliceOp = origOp.source().getDefiningOp<IE::SliceOp>();
    if (producerSliceOp == nullptr) {
        return mlir::failure();
    }

    auto finalOffsets = parseIntArrayAttr<int64_t>(producerSliceOp.static_offsets());
    const auto secondOffsets = parseIntArrayAttr<int64_t>(origOp.static_offsets());
    for (auto i : irange(finalOffsets.size())) {
        finalOffsets[i] += secondOffsets[i];
    }

    const auto finalOffsetsAttr = getIntArrayAttr(getContext(), finalOffsets);
    const auto finalShapeAttr = origOp.static_sizes();
    rewriter.replaceOpWithNewOp<IE::SliceOp>(origOp, producerSliceOp.source(), finalOffsetsAttr, finalShapeAttr);

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::SliceOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<ComposeSlice>(ctx);
}
