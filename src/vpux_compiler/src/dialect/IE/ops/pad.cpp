//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"

using namespace vpux;

namespace {

mlir::FailureOr<SmallVector<int64_t>> extractPads(mlir::Location loc, const mlir::Value& padValue,
                                                  const Optional<mlir::ArrayAttr>& padAttr, vpux::ShapeRef inputShape) {
    if (padAttr.hasValue()) {
        return parseIntArrayAttr<int64_t>(padAttr.getValue());
    } else if (padValue != nullptr) {
        auto padsConst = padValue.getDefiningOp<Const::DeclareOp>();
        if (padsConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for pad");
        }

        auto padValueShape = padValue.getType().cast<mlir::ShapedType>().getShape();
        if (padValueShape.size() != 1 || padValueShape[0] != checked_cast<int64_t>(inputShape.size())) {
            return errorAt(loc, "pad_begin shape is not compatible with input tensor."
                                "The length of the list must be equal to the number of dimensions in the input tensor");
        }

        const auto padContent = padsConst.content();
        return to_small_vector(padContent.getValues<int64_t>());
    }

    return errorAt(loc, "Pads were not provided");
}

}  // namespace

mlir::LogicalResult vpux::IE::PadOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::PadOpAdaptor pad(operands, attrs);
    if (mlir::failed(pad.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = pad.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inType.getShape();

    auto padBegin = extractPads(loc, pad.pads_begin(), pad.pads_begin_attr(), inputShape);
    if (mlir::failed(padBegin)) {
        return mlir::failure();
    }
    const auto padEnd = extractPads(loc, pad.pads_end(), pad.pads_end_attr(), inputShape);
    if (mlir::failed(padEnd)) {
        return mlir::failure();
    }
    if (pad.mode() == IE::PadMode::CONSTANT && pad.pad_value() == nullptr && !pad.pad_value_attr().hasValue()) {
        return errorAt(loc, "pad_mode is CONSTANT but pad_value hasn't provided");
    }

    const auto newType = inType.pad(ShapeRef(padBegin.getValue()), ShapeRef(padEnd.getValue()));
    const auto newTensorType = newType.cast<mlir::RankedTensorType>();
    inferredReturnShapes.emplace_back(newTensorType.getShape(), newTensorType.getElementType());

    return mlir::success();
}

namespace {

//
// ConvertConstToAttr
//

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::PadOp> {
public:
    using mlir::OpRewritePattern<IE::PadOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::PadOp padOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::PadOp padOp, mlir::PatternRewriter& rewriter) const {
    if (padOp.pads_begin_attr().hasValue() || padOp.pads_end_attr().hasValue() || padOp.pad_value_attr().hasValue()) {
        return mlir::failure();
    }

    const auto inType = padOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inType.getShape();

    // convert pads_begin

    auto padsBegin = extractPads(padOp.getLoc(), padOp.pads_begin(), padOp.pads_begin_attr(), inputShape);
    if (mlir::failed(padsBegin)) {
        return mlir::failure();
    }
    const auto padsBeginAttr = getIntArrayAttr(padOp.getContext(), padsBegin.getValue());

    // convert pads_end

    auto padsEnd = extractPads(padOp.getLoc(), padOp.pads_end(), padOp.pads_end_attr(), inputShape);
    if (mlir::failed(padsEnd)) {
        return mlir::failure();
    }
    const auto padsEndAttr = getIntArrayAttr(padOp.getContext(), padsEnd.getValue());

    // convert pad_value

    if (padOp.pad_value() != nullptr) {
        const auto padValueType = padOp.pad_value().getType().cast<mlir::ShapedType>();
        if (padValueType.getRank() != 0) {
            return errorAt(padOp.getLoc(), "'pad_value' must be scalar, got tensor with rank: {0}",
                           padValueType.getRank());
        }

        auto padValueConst = padOp.pad_value().getDefiningOp<Const::DeclareOp>();
        if (padValueConst == nullptr) {
            return errorAt(padOp.getLoc(), "Only constant input is supported for 'pad_value'");
        }

        const auto padValueContent = padValueConst.content();
        if (!padValueContent.isSplat()) {
            return errorAt(padOp.getLoc(), "Only splat input is supported for 'pad_value'");
        }

        const auto padValue = padValueContent.getSplatValue<float>();
        const auto padValueAttr = getFPAttr(padOp.getContext(), padValue);

        rewriter.replaceOpWithNewOp<IE::PadOp>(padOp, padOp.input(), nullptr, nullptr, nullptr, padsBeginAttr,
                                               padsEndAttr, padValueAttr, padOp.mode());
    } else {
        rewriter.replaceOpWithNewOp<IE::PadOp>(padOp, padOp.input(), nullptr, nullptr, nullptr, padsBeginAttr,
                                               padsEndAttr, nullptr, padOp.mode());
    }
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::PadOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::PadOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (input().getType() == output().getType()) {
        return input();
    }

    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        if (mode() == IE::PadMode::CONSTANT) {
            if (pads_begin_attr().hasValue() && pads_end_attr().hasValue() && pad_value_attr().hasValue()) {
                if (pad_value_attr()->convertToDouble() == 0.0) {
                    const auto padsBefore = Shape(parseIntArrayAttr<int64_t>(pads_begin_attr().getValue()));
                    const auto padsAfter = Shape(parseIntArrayAttr<int64_t>(pads_end_attr().getValue()));

                    return attr.padWithZero(padsBefore, padsAfter);
                }
            }
        }
    }

    return nullptr;
}
