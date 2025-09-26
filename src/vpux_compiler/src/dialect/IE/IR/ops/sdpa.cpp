//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/type/float16.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::SDPAOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::SDPAOpAdaptor sdpa(operands, attrs, prop);
    if (mlir::failed(sdpa.verify(loc))) {
        return mlir::failure();
    }
    const auto inQType = mlir::cast<vpux::NDTypeInterface>(sdpa.getInputQ().getType());
    const auto inQShape = inQType.getShape().raw();
    const auto rank = inQType.getShape().size();

    const auto inKType = mlir::cast<vpux::NDTypeInterface>(sdpa.getInputK().getType());
    const auto inKShape = inKType.getShape().raw();

    const auto inVType = mlir::cast<vpux::NDTypeInterface>(sdpa.getInputV().getType());
    const auto inVShape = inVType.getShape().raw();

    const auto isTransposedV = inKShape[rank - 2] != inVShape[rank - 2];
    const auto Ev = isTransposedV ? inVShape[rank - 2] : inVShape[rank - 1];
    SmallVector<int64_t> outShape(inQShape.begin(), inQShape.end());
    outShape[rank - 1] = Ev;
    inferredReturnShapes.emplace_back(outShape, inQType.getElementType());

    return mlir::success();
}

//
// CreateCausalAttentionMask
//

namespace {

class CreateCausalAttentionMask final : public mlir::OpRewritePattern<IE::SDPAOp> {
public:
    using mlir::OpRewritePattern<IE::SDPAOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::SDPAOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult CreateCausalAttentionMask::matchAndRewrite(IE::SDPAOp origOp,
                                                               mlir::PatternRewriter& rewriter) const {
    if (!origOp.getCausal()) {
        return mlir::failure();
    }

    auto queryShape = origOp.getInputQ().getType().getShape();
    auto keyShape = origOp.getInputK().getType().getShape();

    if (queryShape.size() < 2 || keyShape.size() < 2) {
        return errorAt(origOp, "Rank of Query ({0}) and Key ({1}) tensors must be at least 2", queryShape.size(),
                       keyShape.size());
    }

    auto targetSeqLen = queryShape[keyShape.size() - 2];
    auto sourceSeqLen = keyShape[keyShape.size() - 2];
    auto attentionMaskDims = SmallVector<int64_t>{targetSeqLen, sourceSeqLen};

    auto data = SmallVector<type::float16>(targetSeqLen * sourceSeqLen);

    // Fill the upper triangle matrix with -inf
    auto minusInf = type::float16(-std::numeric_limits<float>::infinity());
    for (int64_t h = 0; h < targetSeqLen; h++) {
        for (int64_t w = h + 1; w < sourceSeqLen; w++) {
            data[h * sourceSeqLen + w] = minusInf;
        }
    }

    // Create an attention mask constant for the causal case
    auto ctx = rewriter.getContext();
    auto attentionMaskType = mlir::RankedTensorType::get(attentionMaskDims, mlir::Float16Type::get(ctx));

    auto denseElementVal = Const::createConstContent(attentionMaskType, ArrayRef(data));
    auto causalMask = rewriter.create<Const::DeclareOp>(origOp.getLoc(), attentionMaskType,
                                                        Const::ContentAttr::get(denseElementVal));

    rewriter.replaceOpWithNewOp<IE::SDPAOp>(origOp, origOp.getInputQ(), origOp.getInputK(), origOp.getInputV(),
                                            causalMask, origOp.getInputScale(), /*causal*/ nullptr);

    return mlir::success();
}

}  // namespace

void vpux::IE::SDPAOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<CreateCausalAttentionMask>(context);
}
