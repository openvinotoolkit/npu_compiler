//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ConvertOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ConvertOpAdaptor cvt(operands, attrs, prop);
    if (mlir::failed(cvt.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::RankedTensorType>(cvt.getInput().getType());
    const auto dstElemType = cvt.getDstElemType();

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType, inType.getEncoding());
    return mlir::success();
}

bool vpux::IE::ConvertOp::areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        return false;
    }

    const auto input = mlir::cast<vpux::NDTypeInterface>(inputs.front());
    const auto output = mlir::cast<vpux::NDTypeInterface>(outputs.front());

    return input.getShape() == output.getShape();
}

namespace {

#include <vpux/compiler/dialect/IE/convert.hpp.inc>

// Fuses two consecutive converts only when the intermediate conversion is redundant, i.e. it does not change the
// value: either the inner convert targets the input type, or the outer convert targets the intermediate type.
// Lossy intermediates (float -> int, f32 -> f16, ...) must be preserved; fusing them away drops the truncation or
// rounding performed by the inner conversion.
class FuseRedundantConverts final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    explicit FuseRedundantConverts(mlir::MLIRContext* ctx): mlir::OpRewritePattern<IE::ConvertOp>(ctx) {
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const final {
        const auto innerOp = origOp->getOperand(0).getDefiningOp<IE::ConvertOp>();
        if (innerOp == nullptr || !innerOp->hasOneUse()) {
            return mlir::failure();
        }
        const auto inputType = mlir::cast<vpux::NDTypeInterface>(innerOp->getOperand(0).getType()).getElementType();
        const auto midType = mlir::cast<vpux::NDTypeInterface>(innerOp->getResult(0).getType()).getElementType();
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType()).getElementType();
        if (midType != inputType && midType != outputType) {
            return mlir::failure();
        }
        rewriter.replaceOpWithNewOp<IE::ConvertOp>(origOp, innerOp->getOperand(0), origOp.getDstElemType());
        return mlir::success();
    }
};

}  // namespace

void vpux::IE::ConvertOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext*) {
    patterns.add<FuseRedundantConverts>(patterns.getContext());
    populateWithGenerated(patterns);
}

void vpux::IE::registerConvertOpRewriters(RewriterRegistry& registry) {
    registry.registerRewriter<FuseRedundantConverts>("fuse-converts");
}

mlir::OpFoldResult vpux::IE::ConvertOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    VPUX_THROW_UNLESS(operands.size() == 1, "Expected exactly one operand, but got {0}", operands.size());

    if (auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        return attr.transform().castElemType(getDstElemType()).get();
    }

    return nullptr;
}

//
// verify
//

mlir::LogicalResult vpux::IE::ConvertOp::verify() {
    const auto inTy = mlir::cast<vpux::NDTypeInterface>(getInput().getType());
    const auto outTy = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());

    if (inTy.getShape().isDynamic()) {
        if (!mlir::isa<Core::BoundedTensorType>(inTy)) {
            return errorAt(*this, "Missed bounds for input with dynamic dims");
        }
    }
    if (outTy.getShape().isDynamic()) {
        if (!mlir::isa<Core::BoundedTensorType>(outTy)) {
            return errorAt(*this, "Missed bounds for output with dynamic dims");
        }
    }

    return mlir::success();
}

//
// ShaveCodeGenSupportedOpInterface
//

bool vpux::IE::ConvertOp::shouldJITCompile() {
    auto inType = getInput().getType().getElementType();
    auto outType = getOutput().getType().getElementType();

    if (!vpux::ShaveCodeGen::hasOnlySupportedTypes(*this)) {
        return false;
    }
    if ((inType.getIntOrFloatBitWidth() == outType.getIntOrFloatBitWidth()) && inType.isIntOrIndex() &&
        outType.isIntOrIndex()) {
        return false;
    }
    if (config::getCompilationMode(*this) == config::CompilationMode::ReferenceSW) {
        return true;
    }

    return !isConvertSupportedOnDMA<IE::ConvertOp>(*this);
}

mlir::LogicalResult vpux::IE::ConvertOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                           mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    reifiedReturnShapes.emplace_back(reifyTrivialTensor(builder, getInput(), getLoc()));
    return mlir::success();
}
