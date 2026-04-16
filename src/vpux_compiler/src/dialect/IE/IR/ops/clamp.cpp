//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/custom_float.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::IE::ClampOp::verify() {
    auto inElemType = mlir::cast<vpux::NDTypeInterface>(getInput().getType()).getElementType();
    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(inElemType)) {
        return errorAt(*this, "Per-axis quantized type is not supported. Got: {0}", inElemType);
    }

    const auto minVal = getMinAttr().getValueAsDouble();
    const auto maxVal = getMaxAttr().getValueAsDouble();
    if (minVal > maxVal) {
        return errorAt(*this, "ClampOp {0} has invalid minAttr {1} and maxAttr {2}", getLoc(), minVal, maxVal);
    }

    return mlir::success();
}

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::ClampOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ClampOpAdaptor clamp(operands, attrs, prop);
    if (mlir::failed(clamp.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::RankedTensorType>(clamp.getInput().getType());
    const auto outDesc = vpux::getTensorAttr(inType);
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType(), outDesc);

    return mlir::success();
}

namespace {

template <typename T>
std::pair<double, double> getTypeNumericRange() {
    return {checked_cast<double>(std::numeric_limits<T>::lowest()),
            checked_cast<double>(std::numeric_limits<T>::max())};
}

//
// Clamp Attr to Data Type Range
//

class ClampAttrToDataTypeRange final : public mlir::OpRewritePattern<IE::ClampOp> {
public:
    using mlir::OpRewritePattern<IE::ClampOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ClampOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ClampAttrToDataTypeRange::matchAndRewrite(IE::ClampOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    const auto minVal = origOp.getMinAttr().getValueAsDouble();
    const auto maxVal = origOp.getMaxAttr().getValueAsDouble();

    const auto inType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto elemType = inType.getElementType();

    double typeLowest = 0.0;
    double typeMax = 0.0;

    if (elemType.isF16()) {
        std::tie(typeLowest, typeMax) = getTypeNumericRange<vpux::type::float16>();
    } else if (elemType.isBF16()) {
        std::tie(typeLowest, typeMax) = getTypeNumericRange<vpux::type::bfloat16>();
    } else if (elemType.isF32()) {
        std::tie(typeLowest, typeMax) = getTypeNumericRange<float>();
    } else if (elemType.isSignedInteger(32)) {
        std::tie(typeLowest, typeMax) = getTypeNumericRange<int32_t>();
    } else if (elemType.isSignedInteger(16)) {
        std::tie(typeLowest, typeMax) = getTypeNumericRange<int16_t>();
    } else if (elemType.isUnsignedInteger(32)) {
        std::tie(typeLowest, typeMax) = getTypeNumericRange<uint32_t>();
    } else if (elemType.isUnsignedInteger(16)) {
        std::tie(typeLowest, typeMax) = getTypeNumericRange<uint16_t>();
    } else {
        return mlir::failure();
    }

    const auto isOutOfRange = [typeLowest, typeMax](double value) {
        return value < typeLowest || value > typeMax;
    };

    if (!isOutOfRange(minVal) && !isOutOfRange(maxVal)) {
        return mlir::failure();
    }

    if (minVal > typeMax || maxVal < typeLowest) {
        Logger::global().warning("ClampOp operation at location {0} has a value range from {1} to {2}, which exceeds "
                                 "the data type range [{3}, {4}]. The output values are adjusted.",
                                 origOp.getLoc(), minVal, maxVal, typeLowest, typeMax);
    }

    const auto newMin = std::clamp(minVal, typeLowest, typeMax);
    const auto newMax = std::clamp(maxVal, typeLowest, typeMax);
    const auto minAttr = getFPAttr(origOp.getContext(), newMin);
    const auto maxAttr = getFPAttr(origOp.getContext(), newMax);

    rewriter.replaceOpWithNewOp<IE::ClampOp>(origOp, origOp.getInput(), minAttr, maxAttr);
    return mlir::success();
}

//
// Fuse Clamps
//

class FuseClamps final : public mlir::OpRewritePattern<IE::ClampOp> {
public:
    using mlir::OpRewritePattern<IE::ClampOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ClampOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseClamps::matchAndRewrite(IE::ClampOp origOp, mlir::PatternRewriter& rewriter) const {
    auto parentOp = origOp.getInput().getDefiningOp<IE::ClampOp>();
    if (parentOp == nullptr) {
        return mlir::failure();
    }

    if (!parentOp.getResult().hasOneUse()) {
        return mlir::failure();
    }

    const auto minParentOp = parentOp.getMinAttr().getValueAsDouble();
    const auto minOrigOp = origOp.getMinAttr().getValueAsDouble();
    const auto maxParentOp = parentOp.getMaxAttr().getValueAsDouble();
    const auto maxOrigOp = origOp.getMaxAttr().getValueAsDouble();

    const auto newMin = std::max(minParentOp, minOrigOp);
    const auto newMax = std::min(maxParentOp, maxOrigOp);

    rewriter.replaceOpWithNewOp<IE::ClampOp>(origOp, parentOp.getInput(), getFPAttr(rewriter, newMin),
                                             getFPAttr(rewriter, newMax));
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::ClampOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<FuseClamps>(ctx);
    patterns.add<ClampAttrToDataTypeRange>(ctx);
}
