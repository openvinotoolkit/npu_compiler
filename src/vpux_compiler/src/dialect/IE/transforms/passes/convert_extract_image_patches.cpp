//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTEXTRACTIMAGEPATCHES
#define GEN_PASS_DEF_CONVERTEXTRACTIMAGEPATCHES
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// ======================================================================================
// ConvertToReduceSumRewriter
class ConvertToReduceSumRewriter final : public mlir::OpRewritePattern<IE::ExtractImagePatchesOp> {
public:
    ConvertToReduceSumRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExtractImagePatchesOp>(ctx, benefitLow), _log(log) {
        setDebugName("ConvertToReduceSumRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExtractImagePatchesOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isExtractImagePatchesJustTranspose(IE::ExtractImagePatchesOp op, Logger log) {
    if (op.getSizes() == nullptr || op.getStrides() == nullptr || op.getRates() == nullptr ||
        op.getAutoPadAttr() == nullptr) {
        return false;
    }

    const auto dataShape = getShape(op.getData());
    const auto sizes = parseIntArrayAttr<int64_t>(op.getSizes());
    const auto strides = parseIntArrayAttr<int64_t>(op.getStrides());
    const auto rates = parseIntArrayAttr<int64_t>(op.getRates());

    // Check that the ExtractImagePatrches does only a transpose
    if (sizes.size() != 2 || sizes[vpux::Dims4D::Kernel::Y.ind()] != 1 ||
        sizes[vpux::Dims4D::Kernel::X.ind()] != dataShape[vpux::Dims4D::Act::W]) {
        return false;
    }
    if (strides.size() != 2 ||
        (strides[vpux::Dims4D::Kernel::Y.ind()] != 1 || strides[vpux::Dims4D::Kernel::X.ind()] != 1)) {
        return false;
    }
    if (rates.size() != 2 || (rates[vpux::Dims4D::Kernel::Y.ind()] != 1 || rates[vpux::Dims4D::Kernel::X.ind()] != 1)) {
        return false;
    }
    if (op.getAutoPad() != IE::PadType::VALID) {
        return false;
    }
    if (dataShape[vpux::Dims4D::Act::C] != 1) {
        return false;
    }
    log.trace("ExtractImagePatches op is equivalent with a transpose op. - '{0}'", op->getLoc());
    return true;
}

mlir::LogicalResult replaceExtractImagePatchesWithTranspose(IE::ExtractImagePatchesOp& op,
                                                            mlir::PatternRewriter& rewriter, Logger log) {
    mlir::MLIRContext* ctx = op->getContext();
    // The equivalent permutation order
    log.trace("ExtractImagePatches op is replaced with a Transpose op - '{0}'", op->getLoc());
    const auto orderOutputAttr = mlir::AffineMapAttr::get(vpux::DimsOrder::NWHC.toAffineMap(ctx));
    auto newOp = rewriter.replaceOpWithNewOp<IE::TransposeOp>(op, op.getType(), op.getData(), nullptr, orderOutputAttr);
    extendOpLoc(newOp, "as_transpose");
    return mlir::success();
}

mlir::LogicalResult ConvertToReduceSumRewriter::matchAndRewrite(IE::ExtractImagePatchesOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // Check if ExtractImagePatches is equivalent with a single Transpose
    if (!isExtractImagePatchesJustTranspose(origOp, _log)) {
        return mlir::failure();
    }
    // Check if one of the following patterns appears:
    // 1. ReduceSum -> ExtractImagePatches -> Tranpose -> ReduceSum
    // 2. ReduceSum -> ExtractImagePatches -> ReduceSum
    auto aboveReduceSumOp = origOp.getData().getDefiningOp<IE::ReduceSumOp>();
    if (aboveReduceSumOp != nullptr) {
        // The ReduceSum op should have only one consumer
        if (!aboveReduceSumOp.getResult().hasOneUse()) {
            return replaceExtractImagePatchesWithTranspose(origOp, rewriter, _log);
        }
        // The ExtractImagePatches op should have only one consumer
        if (!origOp.getResult().hasOneUse()) {
            return replaceExtractImagePatchesWithTranspose(origOp, rewriter, _log);
        }

        auto transposeOp = mlir::dyn_cast<IE::TransposeOp>(*(origOp.getOutput().getUsers().begin()));
        vpux::IE::ReduceSumOp belowReduceSumOp;
        if (transposeOp != nullptr) {
            // The Transpose op should have only one consumer
            if (!transposeOp.getResult().hasOneUse()) {
                return replaceExtractImagePatchesWithTranspose(origOp, rewriter, _log);
            }
            belowReduceSumOp = mlir::dyn_cast<IE::ReduceSumOp>(*(transposeOp.getOutput().getUsers().begin()));
        } else {
            belowReduceSumOp = mlir::dyn_cast<IE::ReduceSumOp>(*(origOp.getOutput().getUsers().begin()));
        }

        if (belowReduceSumOp == nullptr) {
            return replaceExtractImagePatchesWithTranspose(origOp, rewriter, _log);
        }
        _log.trace("Found ReduceSum -> ExtractImagePatches -> [Tranpose] -> ReduceSum subgraph. - '{0}",
                   origOp->getLoc());

        // Additional checks for aboveReduceSumOp
        if (!aboveReduceSumOp.getKeepDims() || !aboveReduceSumOp.getAxesValue().has_value()) {
            return replaceExtractImagePatchesWithTranspose(origOp, rewriter, _log);
        }

        auto aboveReduceSumOpAxes = parseIntArrayAttr<int64_t>(aboveReduceSumOp.getAxesValue().value());
        if (aboveReduceSumOpAxes.size() > 1 || aboveReduceSumOpAxes[0] != vpux::Dims4D::Act::C.ind()) {
            return replaceExtractImagePatchesWithTranspose(origOp, rewriter, _log);
        }

        // Additional checks for belowReduceSumOp
        if (!belowReduceSumOp.getAxesValue().has_value()) {
            return replaceExtractImagePatchesWithTranspose(origOp, rewriter, _log);
        }

        auto belowReduceSumOpAxes = parseIntArrayAttr<int64_t>(belowReduceSumOp.getAxesValue().value());
        if (transposeOp == nullptr) {
            if (belowReduceSumOpAxes.size() > 1 || belowReduceSumOpAxes[0] != vpux::Dims4D::Act::C.ind()) {
                return replaceExtractImagePatchesWithTranspose(origOp, rewriter, _log);
            }
        } else {
            const auto transposeOpPerm = vpux::DimsOrder::fromAffineMap(transposeOp.getOrderValue().value());
            if (transposeOpPerm != vpux::DimsOrder::NHWC || belowReduceSumOpAxes.size() > 1 ||
                belowReduceSumOpAxes[0] != vpux::Dims4D::Act::W.ind()) {
                return replaceExtractImagePatchesWithTranspose(origOp, rewriter, _log);
            }
        }
        _log.trace("Confirmed that the subgraph satisfies the rest of constrains. - '{0}", origOp->getLoc());

        // Replace the subgraph with the equivalent ReduceSum op
        aboveReduceSumOpAxes.push_back(vpux::Dims4D::Act::W.ind());
        // Create the array attribute containing the fused reduction axes
        mlir::MLIRContext* ctx = origOp->getContext();
        auto axesAttr = getIntArrayAttr(ctx, ArrayRef(aboveReduceSumOpAxes));
        auto newReduceSumOp = rewriter.create<IE::ReduceSumOp>(takeOpLoc(origOp, "reduce_sum_in"),
                                                               aboveReduceSumOp.getInput(), nullptr, axesAttr, nullptr);
        mlir::SmallVector<int64_t> unsqueezeAxis{vpux::Dims4D::Act::H.ind()};
        if (belowReduceSumOp.getKeepDims()) {
            unsqueezeAxis.push_back(belowReduceSumOpAxes[0]);
        }
        const auto unsqueezeAxesAttr = getIntArrayAttr(getContext(), unsqueezeAxis);
        rewriter.replaceOpWithNewOp<IE::UnsqueezeOp>(belowReduceSumOp, newReduceSumOp, nullptr, unsqueezeAxesAttr);
        return mlir::success();
    }

    return replaceExtractImagePatchesWithTranspose(origOp, rewriter, _log);
}

// ======================================================================================
// ConvertToSliceConcatRewriter
class ConvertToSliceConcatRewriter final : public mlir::OpRewritePattern<IE::ExtractImagePatchesOp> {
public:
    ConvertToSliceConcatRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ExtractImagePatchesOp>(ctx, benefitLow), _log(log) {
        setDebugName("ConvertToSliceConcatRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExtractImagePatchesOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertToSliceConcatRewriter::matchAndRewrite(IE::ExtractImagePatchesOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // Check if ExtractImagePatches is equivalent with with NxSlice->Concat sequence
    if (origOp.getSizes() == nullptr || origOp.getStrides() == nullptr || origOp.getRates() == nullptr ||
        origOp.getAutoPadAttr() == nullptr) {
        return mlir::failure();
    }

    const auto dataShape = getShape(origOp.getData());
    const auto sizes = parseIntArrayAttr<int64_t>(origOp.getSizes());
    const auto strides = parseIntArrayAttr<int64_t>(origOp.getStrides());
    const auto rates = parseIntArrayAttr<int64_t>(origOp.getRates());

    // Check that the ExtractImagePatrches does only a transpose
    if (sizes.size() != 2 || sizes[vpux::Dims4D::Kernel::Y.ind()] == 1 ||
        sizes[vpux::Dims4D::Kernel::Y.ind()] >= dataShape[vpux::Dims4D::Act::H] ||
        sizes[vpux::Dims4D::Kernel::X.ind()] != dataShape[vpux::Dims4D::Act::W]) {
        return mlir::failure();
    }
    if (strides.size() != 2 ||
        (strides[vpux::Dims4D::Kernel::Y.ind()] != 1 || strides[vpux::Dims4D::Kernel::X.ind()] != 1)) {
        return mlir::failure();
    }
    if (rates.size() != 2 || (rates[vpux::Dims4D::Kernel::Y.ind()] != 1 || rates[vpux::Dims4D::Kernel::X.ind()] != 1)) {
        return mlir::failure();
    }
    if (origOp.getAutoPad() != IE::PadType::VALID) {
        return mlir::failure();
    }
    if (dataShape[vpux::Dims4D::Act::C] != 1) {
        return mlir::failure();
    }
    _log.trace("ExtractImagePatches op is equivalent with nxSliceOp -> Concat. - '{0}'", origOp->getLoc());

    // Check if one of the following pattern is met:
    // 1. ExtractImagePatches -> Transpose-> AffineReshape
    // 2. ExtractImagePatches -> Transpose
    // If ExtractImagePatches -> Transpose-> AffineReshape pattern + all required constrains are met and:
    // If pattern 1 is met, do the rewrite:
    // ExtractImagePatches -> Transpose-> AffineReshape  =>  NxSlice -> Concat
    // If pattern 2 is met, do the rewrite:
    // ExtractImagePatches -> Transpose                  =>  NxSlice -> Concat -> AffineReshape
    if (!origOp.getResult().hasOneUse()) {
        return mlir::failure();
    }

    auto transposeOp = mlir::dyn_cast<IE::TransposeOp>(*(origOp.getOutput().getUsers().begin()));
    if (transposeOp == nullptr) {
        return mlir::failure();
    }

    const auto origPerm = vpux::DimsOrder::fromAffineMap(transposeOp.getOrderValue().value());
    if (origPerm != vpux::DimsOrder::NHWC) {
        return mlir::failure();
    }
    _log.trace("ExtractImagePatches -> Transpose pattern with all required constrains satisfied met. - '{0}'",
               origOp.getLoc());

    const auto dataHeight = dataShape[vpux::Dims4D::Act::H];
    const auto sizesHeight = (parseIntArrayAttr<int64_t>(origOp.getSizes()))[vpux::Dims4D::Kernel::Y.ind()];
    const auto numberOfSlices = checked_cast<size_t>(dataHeight - sizesHeight + 1);
    mlir::SmallVector<mlir::Value> sliceOpValues;

    if (transposeOp.getOutput().getUsers().empty()) {
        return mlir::failure();
    }
    auto affineReshapeOp = mlir::dyn_cast<IE::AffineReshapeOp>(*(transposeOp.getOutput().getUsers().begin()));

    // Backup function to do the corresponding rewrite of pattern 2
    auto rewriteExtractImagePatchesTranspose = [&](bool suitableReshape) -> mlir::LogicalResult {
        mlir::MLIRContext* ctx = origOp->getContext();
        for (size_t idx = 0; idx < numberOfSlices; idx++) {
            auto staticOffsets = mlir::SmallVector<int64_t>(dataShape.size(), 0);
            staticOffsets[vpux::Dims4D::Act::H.ind()] = idx;
            mlir::SmallVector<int64_t> staticSizes = to_small_vector(dataShape.raw());
            staticSizes[vpux::Dims4D::Act::H.ind()] = sizesHeight;
            auto sliceOp = rewriter.create<IE::SliceOp>(takeOpLoc(origOp, StringLiteral("slice_{0}"), idx),
                                                        origOp.getData(), getIntArrayAttr(ctx, staticOffsets),
                                                        getIntArrayAttr(ctx, staticSizes));
            sliceOpValues.push_back(sliceOp.getResult());
        }
        if (suitableReshape) {
            rewriter.replaceOpWithNewOp<IE::ConcatOp>(affineReshapeOp, sliceOpValues, vpux::Dims4D::Act::C);
        } else {
            auto concatOp = rewriter.create<IE::ConcatOp>(takeOpLoc(origOp, "out_reshape"), sliceOpValues,
                                                          vpux::Dims4D::Act::C);

            const auto transposeShape = transposeOp.getType().getShape();
            const auto transposeShapeAttr = getIntArrayAttr(ctx, transposeShape);

            rewriter.replaceOpWithNewOp<IE::ReshapeOp>(transposeOp, concatOp, nullptr, false, transposeShapeAttr);
        }
        return mlir::success();
    };

    auto isSupportedReshape = [&]() -> bool {
        // Check if pattern 1 is met and do the rewrite
        if (affineReshapeOp != nullptr) {
            // Check that the Transpose op just 1 child
            if (!transposeOp.getResult().hasOneUse()) {
                return false;
            }
            const auto affineReshapeDimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshapeOp.getDimMappingAttr());
            const auto affineReshapeShapeValue = parseIntArrayAttr<int64_t>(affineReshapeOp.getShapeValueAttr());
            // AffineReshape conditions
            if (affineReshapeDimMapping.size() != 4) {
                return false;
            }
            if (affineReshapeDimMapping[vpux::Dims4D::Act::N.ind()].size() != 1 ||
                affineReshapeDimMapping[vpux::Dims4D::Act::N.ind()][0] != 0) {
                return false;
            }
            if (affineReshapeDimMapping[vpux::Dims4D::Act::C.ind()].size() != 1 ||
                affineReshapeDimMapping[vpux::Dims4D::Act::C.ind()][0] != 1) {
                return false;
            }
            if (affineReshapeDimMapping[vpux::Dims4D::Act::H.ind()].size() != 1 ||
                affineReshapeDimMapping[vpux::Dims4D::Act::H.ind()][0] != 1) {
                return false;
            }
            if (affineReshapeDimMapping[vpux::Dims4D::Act::W.ind()].size() != 2 ||
                affineReshapeDimMapping[vpux::Dims4D::Act::W.ind()][0] != 2 ||
                affineReshapeDimMapping[vpux::Dims4D::Act::W.ind()][1] != 3) {
                return false;
            }

            if (affineReshapeShapeValue.size() != 4) {
                return false;
            }
            if (affineReshapeShapeValue[vpux::Dims4D::Act::N.ind()] != 1) {
                return false;
            }
            if (affineReshapeShapeValue[vpux::Dims4D::Act::C.ind()] != checked_cast<int64_t>(numberOfSlices)) {
                return false;
            }
            if (affineReshapeShapeValue[vpux::Dims4D::Act::H.ind()] != sizes[vpux::Dims4D::Kernel::Y.ind()]) {
                return false;
            }
            if (affineReshapeShapeValue[vpux::Dims4D::Act::W.ind()] != sizes[vpux::Dims4D::Kernel::X.ind()]) {
                return false;
            }
            _log.trace("All conditions of ExtractImagePatches -> Transpose-> AffineReshape pattern were met. - '{0}'",
                       origOp->getLoc());
            return true;
        }
        return false;
    };
    return rewriteExtractImagePatchesTranspose(isSupportedReshape());
}

//
// ConvertExtractImagePatchesPass
//

class ConvertExtractImagePatchesPass final :
        public IE::impl::ConvertExtractImagePatchesBase<ConvertExtractImagePatchesPass> {
public:
    explicit ConvertExtractImagePatchesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertExtractImagePatchesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertToReduceSumRewriter>(&ctx, _log);
    patterns.add<ConvertToSliceConcatRewriter>(&ctx, _log);
    IE::ReshapeOp::getCanonicalizationPatterns(patterns, &ctx);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertExtractImagePatchesPass
//
std::unique_ptr<mlir::Pass> vpux::IE::createConvertExtractImagePatchesPass(Logger log) {
    return std::make_unique<ConvertExtractImagePatchesPass>(log);
}
