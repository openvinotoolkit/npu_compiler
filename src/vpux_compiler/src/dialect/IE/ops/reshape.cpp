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

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/PatternMatch.h>

#include <numeric>

using namespace vpux;

//
// getOutShape
//

namespace {

mlir::FailureOr<SmallVector<int64_t>> getOutShape(IE::ReshapeOpAdaptor reshape, mlir::Location loc) {
    if (reshape.shape() != nullptr && reshape.shape_value() != nullptr) {
        return errorAt(loc, "Ambiguous shape representation");
    }
    if (reshape.shape() == nullptr && reshape.shape_value() == nullptr) {
        return errorAt(loc, "Missed shape representation");
    }

    if (reshape.shape_value() != nullptr) {
        return parseIntArrayAttr<int64_t>(reshape.shape_value());
    }

    auto shapeConst = reshape.shape().getDefiningOp<Const::DeclareOp>();
    if (shapeConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for shape");
    }

    const auto shapeContent = shapeConst.content();
    auto shapeVec = to_small_vector(shapeContent.getValues<int64_t>());

    const auto specialZero = reshape.special_zero();

    const auto zeroDims = std::count_if(shapeVec.begin(), shapeVec.end(), [](int64_t v) {
        return v == 0;
    });
    const auto negativeDims = std::count_if(shapeVec.begin(), shapeVec.end(), [](int64_t v) {
        return v == -1;
    });

    if (negativeDims > 1) {
        return errorAt(loc, "Shape can not contain more than 1 negative value");
    }

    if (!(zeroDims != 0 && specialZero) && negativeDims == 0) {
        return shapeVec;
    } else {
        const auto inShape = to_small_vector(reshape.input().getType().cast<mlir::ShapedType>().getShape());

        auto dividend = std::accumulate(inShape.begin(), inShape.end(), int64_t(1), std::multiplies<int64_t>());

        for (size_t i = 0; i < shapeVec.size(); ++i) {
            auto& v = shapeVec[i];

            if (v == 0 && specialZero) {
                if (i < inShape.size()) {
                    v = inShape[i];
                } else {
                    return errorAt(loc, "Shape value at '{0}' is out of range '{1}'", i, inShape.size());
                }
            }

            if (v > 0) {
                if (dividend % v != 0) {
                    return errorAt(loc, "Shape value at '{0}' ('{1}') is invalid", i, v);
                }

                dividend /= v;
            }
        }

        if (negativeDims > 0) {
            const auto negIt = std::find(shapeVec.begin(), shapeVec.end(), -1);
            VPUX_THROW_UNLESS(negIt != shapeVec.end(), "Shape vector broken");

            *negIt = dividend;
        }

        return shapeVec;
    }
}

}  // namespace

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::ReshapeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::ReshapeOpAdaptor reshape(operands, attrs);
    if (mlir::failed(reshape.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = getOutShape(reshape, loc);
    if (mlir::failed(outShape)) {
        return mlir::failure();
    }

    const auto inType = reshape.input().getType().cast<mlir::RankedTensorType>();

    const auto outDesc = IE::getTensorAttr(ctx, DimsOrder::fromNumDims(outShape->size()), IE::getMemorySpace(inType));

    inferredReturnShapes.emplace_back(outShape.getValue(), inType.getElementType(), outDesc);
    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ReshapeOp::fold(ArrayRef<mlir::Attribute> operands) {
    if (input().getType() == output().getType()) {
        return input();
    }

    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return attr.reshape(getShape(output()));
    }

    return nullptr;
}

//
// FuseReshapes
//

namespace {

class FuseReshapes final : public mlir::OpRewritePattern<IE::ReshapeOp> {
public:
    using mlir::OpRewritePattern<IE::ReshapeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseReshapes::matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const {
    auto prevOp = origOp.input().getDefiningOp();
    if (prevOp == nullptr) {
        return mlir::failure();
    }
    if (!mlir::isa<IE::SqueezeOp, IE::UnsqueezeOp, IE::ReshapeOp>(prevOp)) {
        return mlir::failure();
    }

    const auto outputShape = origOp.getType().getShape();
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);

    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, prevOp->getOperand(0), nullptr, false, outputShapeAttr);
    return mlir::success();
}

}  // namespace

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::ReshapeOp> {
public:
    using mlir::OpRewritePattern<IE::ReshapeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const {
    if (origOp.shape_value().hasValue()) {
        return mlir::failure();
    }

    const auto outShape = getOutShape(origOp, origOp->getLoc());
    if (mlir::failed(outShape)) {
        return mlir::failure();
    }

    const auto outShapeAttr = getIntArrayAttr(getContext(), outShape.getValue());

    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, origOp.input(), nullptr, false, outShapeAttr);
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::ReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.insert<FuseReshapes>(ctx);
    patterns.insert<ConvertConstToAttr>(ctx);
}
