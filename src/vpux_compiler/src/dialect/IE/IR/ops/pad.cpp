//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/utils/pad_extract.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::PadOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::PadOpAdaptor pad(operands, attrs, prop);
    if (mlir::failed(pad.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(pad.getInput().getType());
    const auto inputShape = inType.getShape();

    auto padBegin = IE::extractPads(loc, pad.getPadsBegin(), pad.getPadsBeginAttr(), inputShape);
    if (mlir::failed(padBegin)) {
        return mlir::failure();
    }

    const auto padEnd = IE::extractPads(loc, pad.getPadsEnd(), pad.getPadsEndAttr(), inputShape);
    if (mlir::failed(padEnd)) {
        return mlir::failure();
    }

    if (pad.getMode() == IE::PadMode::CONSTANT && pad.getPadValue() == nullptr && !pad.getPadValueAttr().has_value()) {
        return errorAt(loc, "pad_mode is CONSTANT but pad_value hasn't provided");
    }

    if (!padBegin.value().empty() && !padEnd.value().empty()) {
        const auto newType = inType.pad(ShapeRef(padBegin.value()), ShapeRef(padEnd.value()));
        const auto newTensorType = mlir::cast<mlir::RankedTensorType>(newType);
        inferredReturnShapes.emplace_back(newTensorType.getShape(), newTensorType.getElementType(),
                                          getTensorAttr(newTensorType));
    } else {
        const auto outShape = parseIntArrayAttr<int64_t>(pad.getOutputShapeAttr());
        const auto outBounds = parseIntArrayAttr<int64_t>(pad.getOutputBoundsAttr());

        const auto inType = mlir::cast<mlir::RankedTensorType>(pad.getInput().getType());

        const auto outDesc = vpux::getTensorAttr(ctx, DimsOrder::fromNumDims(outShape.size()),
                                                 vpux::getMemorySpace(inType), BoundsRef(outBounds));

        inferredReturnShapes.emplace_back(outShape, inType.getElementType(), outDesc);
    }
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
    if (padOp.getPadsBeginAttr().has_value() || padOp.getPadsEndAttr().has_value() ||
        padOp.getPadValueAttr().has_value()) {
        return mlir::failure();
    }

    // All inputs are not `Constant`
    const bool padsBeginIsConst = padOp.getPadsBegin().getDefiningOp<Const::DeclareOp>() != nullptr;
    const bool padsEndIsConst = padOp.getPadsEnd().getDefiningOp<Const::DeclareOp>() != nullptr;
    const bool padValueIsConst =
            padOp.getPadValue() && padOp.getPadValue().getDefiningOp<Const::DeclareOp>() != nullptr;

    if (!padsBeginIsConst && !padsEndIsConst) {
        if (padOp.getMode() != vpux::IE::PadMode::CONSTANT) {
            return mlir::failure();
        }
        if (!padValueIsConst) {
            return mlir::failure();
        }
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(padOp.getInput().getType());
    const auto inputShape = inType.getShape();

    // convert pads_begin

    auto padsBegin = IE::extractPads(padOp.getLoc(), padOp.getPadsBegin(), padOp.getPadsBeginAttr(), inputShape);
    if (mlir::failed(padsBegin)) {
        return mlir::failure();
    }
    const auto padsBeginAttr =
            padsBegin.value().empty() ? nullptr : getIntArrayAttr(padOp.getContext(), padsBegin.value());
    const auto padsBeginValue =
            padsBeginAttr ? nullptr : padOp.getPadsBegin();  // in case if pad_begin is a tensor not a constant

    VPUX_THROW_WHEN(padsBeginAttr == nullptr && padsBeginValue == nullptr,
                    "PadOp is malformed: required input 'pads_begin' is not provided at {0}", padOp->getLoc());

    // convert pads_end

    auto padsEnd = IE::extractPads(padOp.getLoc(), padOp.getPadsEnd(), padOp.getPadsEndAttr(), inputShape);
    if (mlir::failed(padsEnd)) {
        return mlir::failure();
    }
    const auto padsEndAttr = padsEnd.value().empty() ? nullptr : getIntArrayAttr(padOp.getContext(), padsEnd.value());
    const auto padsEndValue =
            padsEndAttr ? nullptr : padOp.getPadsEnd();  // in case if pad_end is a tensor not a constant

    VPUX_THROW_WHEN(padsEndAttr == nullptr && padsEndValue == nullptr,
                    "PadOp is malformed: required input 'pads_end' is not provided at {0}", padOp->getLoc());

    // convert pad_value

    if (padOp.getPadValue() != nullptr) {
        const auto padValueType = mlir::cast<mlir::ShapedType>(padOp.getPadValue().getType());
        if (padValueType.getNumElements() != 1) {
            // Cannot convert const to attr: 'pad_value' has more than 1 element
            return mlir::failure();
        }

        auto padValueConst = padOp.getPadValue().getDefiningOp<Const::DeclareOp>();
        if (padValueConst == nullptr) {
            return errorAt(padOp, "Cannot convert const to attr: 'pad_value' is not const");
        }

        const auto padValueContent = padValueConst.getContent();
        if (!padValueContent.isSplat()) {
            return errorAt(padOp, "Cannot convert const to attr: 'pad_value' is not splat const");
        }

        const auto padValue = padValueContent.getSplatValue<float>();
        const auto padValueAttr = getFPAttr(padOp.getContext(), padValue);
        rewriter.replaceOpWithNewOp<IE::PadOp>(padOp, padOp.getInput(), padsBeginValue, padsEndValue, nullptr,
                                               padsBeginAttr, padsEndAttr, padValueAttr, padOp.getMode(),
                                               padOp.getOutputPaddingAttr(), padOp.getInputPaddingAttr(),
                                               padOp.getOutputShapeAttr(), padOp.getOutputBoundsAttr());
    } else {
        rewriter.replaceOpWithNewOp<IE::PadOp>(padOp, padOp.getInput(), padsBeginValue, padsEndValue, nullptr,
                                               padsBeginAttr, padsEndAttr, nullptr, padOp.getMode(),
                                               padOp.getOutputPaddingAttr(), padOp.getInputPaddingAttr(),
                                               padOp.getOutputShapeAttr(), padOp.getOutputBoundsAttr());
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

mlir::OpFoldResult vpux::IE::PadOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    VPUX_THROW_WHEN(operands.empty(), "Wrong number of operands : {0}", operands.size());

    if (const auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        if (getMode() == IE::PadMode::CONSTANT) {
            if (getPadsBeginAttr().has_value() && getPadsEndAttr().has_value() && getPadValueAttr().has_value()) {
                if (getPadValueAttr()->convertToDouble() == 0.0) {
                    const auto padsBefore = Shape(parseIntArrayAttr<int64_t>(getPadsBeginAttr().value()));
                    const auto padsAfter = Shape(parseIntArrayAttr<int64_t>(getPadsEndAttr().value()));

                    return static_cast<Const::ContentAttr>(attr).transform().padWithZero(padsBefore, padsAfter).get();
                }
            }
        }
    }

    return nullptr;
}
