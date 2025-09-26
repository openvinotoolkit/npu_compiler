//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::ReverseSequenceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReverseSequenceOpAdaptor rev(operands, attrs, prop);
    if (mlir::failed(rev.verify(loc))) {
        return mlir::failure();
    }

    const auto dataType = mlir::cast<mlir::ShapedType>(rev.getData().getType());
    const auto dataShape = dataType.getShape();

    if (dataShape.size() < 2) {
        return errorAt(loc, "First input tensor's size should not be less than 2D. Got {0}D tensor", dataShape.size());
    }

    const auto seqShape = getShape(rev.getSeqLength());

    if (seqShape.size() != 1) {
        return errorAt(loc, "Second input tensor should be 1D Tensor. Got {0}D tensor", seqShape.size());
    }

    const auto dataDims = checked_cast<int64_t>(dataShape.size());

    const auto batch_axis = rev.getBatchAxis();

    if (batch_axis >= dataDims || batch_axis < -dataDims) {
        return errorAt(loc, "ReverseSequence Parameter batch axis {0} out of the tensor rank range [{1}, {2}].",
                       batch_axis, -dataDims, dataDims - 1);
    }

    const auto seq_axis = rev.getSeqAxis();

    if (seq_axis >= dataDims || seq_axis < -dataDims) {
        return errorAt(loc, "ReverseSequence Parameter sequence axis {0} out of the tensor rank range [{1}, {2}].",
                       seq_axis, -dataDims, dataDims - 1);
    }

    if (seqShape[Dims4D::Act::N] != dataShape[batch_axis]) {
        return errorAt(loc, "Sequence lengths input size {0} is not equal to batch axis dimension of data input {1}",
                       seqShape[Dims4D::Act::N], dataShape[batch_axis]);
    }

    const auto elementType = dataType.getElementType();
    if (!(elementType.isF16() || elementType.isF32() || elementType.isInteger(8))) {
        return errorAt(loc, "Reverse Sequence only support FP16, FP32, INT8 (I8/U8/SI8) data type");
    }

    inferredReturnShapes.emplace_back(dataShape, elementType);

    return mlir::success();
}

mlir::OpFoldResult vpux::IE::ReverseSequenceOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    VPUX_THROW_UNLESS(operands.size() == 2, "Wrong number of operands : {0}", operands.size());

    const auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[1]);
    if (attr == nullptr || !attr.isSplat()) {
        return nullptr;
    }

    const auto content = static_cast<Const::ContentAttr>(attr).fold();
    return (content.getSplatValue<int32_t>() == 1) ? getData() : nullptr;
}

namespace {
class ConvertIntToFP16 final : public mlir::OpRewritePattern<IE::ReverseSequenceOp> {
public:
    using mlir::OpRewritePattern<IE::ReverseSequenceOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ReverseSequenceOp rsOp, mlir::PatternRewriter& rewriter) const final;
};

class NormalizeAxis final : public mlir::OpRewritePattern<IE::ReverseSequenceOp> {
public:
    using mlir::OpRewritePattern<IE::ReverseSequenceOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ReverseSequenceOp rsOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertIntToFP16::matchAndRewrite(IE::ReverseSequenceOp rsOp,
                                                      mlir::PatternRewriter& rewriter) const {
    const auto dataType = mlir::cast<mlir::ShapedType>(rsOp.getData().getType());

    if (dataType.getElementType().isInteger(8)) {
        auto convertOpBefore = rewriter.create<IE::ConvertOp>(appendLoc(rsOp.getLoc(), "cvt_in"), rsOp.getData(),
                                                              mlir::Float16Type::get(getContext()));
        auto reverseSequenceOp =
                rewriter.create<IE::ReverseSequenceOp>(rsOp.getLoc(), convertOpBefore.getOutput(), rsOp.getSeqLength(),
                                                       rsOp.getSeqAxis(), rsOp.getBatchAxis());

        auto getInputTypeAttr = [&]() -> mlir::TypeAttr {
            auto elementType = mlir::cast<mlir::IntegerType>(dataType.getElementType());
            switch (elementType.getSignedness()) {
            case mlir::IntegerType::SignednessSemantics::Unsigned:
                return mlir::TypeAttr::get(
                        mlir::IntegerType::get(getContext(), 8, mlir::IntegerType::SignednessSemantics::Unsigned));
            case mlir::IntegerType::SignednessSemantics::Signed:
                return mlir::TypeAttr::get(
                        mlir::IntegerType::get(getContext(), 8, mlir::IntegerType::SignednessSemantics::Signed));
            case mlir::IntegerType::SignednessSemantics::Signless:
            default:
                return mlir::TypeAttr::get(
                        mlir::IntegerType::get(getContext(), 8, mlir::IntegerType::SignednessSemantics::Signless));
            }
        };

        mlir::TypeAttr inputTypeAttr = getInputTypeAttr();
        auto outOp = rewriter.replaceOpWithNewOp<IE::ConvertOp>(rsOp, reverseSequenceOp.getOutput(), inputTypeAttr);
        extendOpLoc(outOp, "cvt_out");
        return mlir::success();
    }
    return mlir::failure();
}

mlir::LogicalResult NormalizeAxis::matchAndRewrite(IE::ReverseSequenceOp rsOp, mlir::PatternRewriter& rewriter) const {
    const auto dataShape = getShape(rsOp.getData());
    const auto dataDims = checked_cast<int64_t>(dataShape.size());
    const auto seq_axis = rsOp.getSeqAxis();
    const auto batch_axis = rsOp.getBatchAxis();
    if (seq_axis >= 0 && batch_axis >= 0) {
        return mlir::failure();
    }
    const auto normalized_seq_axis = seq_axis >= 0 ? seq_axis : seq_axis + dataDims;
    const auto normalized_batch_axis = batch_axis >= 0 ? batch_axis : batch_axis + dataDims;
    rewriter.replaceOpWithNewOp<IE::ReverseSequenceOp>(rsOp, rsOp.getData(), rsOp.getSeqLength(), normalized_seq_axis,
                                                       normalized_batch_axis);
    return mlir::success();
}

}  // namespace

void vpux::IE::ReverseSequenceOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                              mlir::MLIRContext* context) {
    patterns.add<NormalizeAxis>(context);
    patterns.add<ConvertIntToFP16>(context);
}
