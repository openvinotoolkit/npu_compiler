//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>
#include <openvino/op/util/slice_plan.hpp>

namespace vpux::IE {
#define GEN_PASS_DECL_RESOLVESTRIDEDSLICE
#define GEN_PASS_DEF_RESOLVESTRIDEDSLICE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ResolveStridedSlicePass
//

class ResolveStridedSlicePass final : public IE::impl::ResolveStridedSliceBase<ResolveStridedSlicePass> {
public:
    explicit ResolveStridedSlicePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class SlicePlanning;

private:
    void safeRunOnFunc() final;
};

//
// SlicePlanning
//

class ResolveStridedSlicePass::SlicePlanning final : public mlir::OpRewritePattern<IE::StridedSliceOp> {
public:
    SlicePlanning(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::StridedSliceOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::StridedSliceOp origOp, mlir::PatternRewriter& rewriter) const final;
    static ov::op::util::SlicePlan getSlicePlan(IE::StridedSliceOp origOp);

private:
    Logger _log;
};

ov::op::util::SlicePlan ResolveStridedSlicePass::SlicePlanning::getSlicePlan(IE::StridedSliceOp origOp) {
    const auto getAxisSetArr = [](mlir::ArrayAttr attr) {
        ov::AxisSet axis_set;

        const auto arr = parseIntArrayAttr<int64_t>(attr);
        for (const auto& p : arr | indexed) {
            if (p.value() == 1) {
                axis_set.emplace(p.index());
            }
        }

        return axis_set;
    };

    VPUX_THROW_UNLESS(origOp.getBeginsAttr().has_value(), "begins_attr is null");
    VPUX_THROW_UNLESS(origOp.getEndsAttr().has_value(), "ends_attr is null");
    VPUX_THROW_UNLESS(origOp.getStridesAttr().has_value(), "strides_attr is null");

    const auto beginsVec = to_std_vector(parseIntArrayAttr<int64_t>(origOp.getBeginsAttr().value()));
    const auto endsVec = to_std_vector(parseIntArrayAttr<int64_t>(origOp.getEndsAttr().value()));
    const auto stridesVec = to_std_vector(parseIntArrayAttr<int64_t>(origOp.getStridesAttr().value()));

    const auto beginMask = getAxisSetArr(origOp.getBeginMask());
    const auto endMask = getAxisSetArr(origOp.getEndMask());
    const auto newAxisMask = getAxisSetArr(origOp.getNewAxisMask());
    const auto shrinkAxisMask = getAxisSetArr(origOp.getShrinkAxisMask());
    const auto ellipsisMask = getAxisSetArr(origOp.getEllipsisMask());

    const auto inDataType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inDataShape = inDataType.getShape();

    return ov::op::util::make_slice_plan(ov::Shape(inDataShape.begin(), inDataShape.end()), beginsVec, endsVec,
                                         stridesVec, beginMask, endMask, newAxisMask, shrinkAxisMask, ellipsisMask);
}

mlir::LogicalResult ResolveStridedSlicePass::SlicePlanning::matchAndRewrite(IE::StridedSliceOp origOp,
                                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE::StridedSlice Operation '{0}'", origOp->getLoc());

    const auto plan = getSlicePlan(origOp);

    auto beginAttr = getIntArrayAttr(getContext(), plan.begins);
    const auto reverseAxis = plan.reverse_axes.to_vector();
    IE::LayerOpInterface newOp;
    if (llvm::any_of(plan.strides, [](int64_t val) {
            return val > 1;
        })) {
        const auto endsAttr = getIntArrayAttr(getContext(), plan.ends);
        const auto stridesAttr = getIntArrayAttr(getContext(), plan.strides);
        const auto zeroesArrayAttr = getIntArrayAttr(getContext(), SmallVector<int64_t>(plan.begins.size(), 0));

        newOp = rewriter.create<IE::StridedSliceOp>(takeOpLoc(origOp, "slice_in"), origOp.getInput(),
                                                    origOp.getBegins(), origOp.getEnds(), origOp.getStrides(),
                                                    beginAttr, endsAttr, stridesAttr, zeroesArrayAttr, zeroesArrayAttr,
                                                    zeroesArrayAttr, zeroesArrayAttr, zeroesArrayAttr);
    } else {
        auto source = origOp.getInput();
        if (!reverseAxis.empty()) {
            VPUX_THROW_UNLESS(reverseAxis.size() == 1, "Currently only support reverse on one axis");
            const auto axis = reverseAxis[0];
            const auto inShape = getShape(source);
            const auto sliceNums = inShape[Dim(axis)];

            SmallVector<mlir::Value> concatInputs;
            SmallVector<int64_t> sliceOffset(inShape.size(), 0);
            SmallVector<int64_t> sliceShape(inShape.begin(), inShape.end());
            sliceShape[axis] = 1;
            for (int64_t i = 0; i < sliceNums; i++) {
                sliceOffset[axis] = i;
                concatInputs.push_back(rewriter.create<IE::SliceOp>(takeOpLoc(origOp, StringLiteral("slice_{0}"), i),
                                                                    source, sliceOffset, sliceShape)
                                               .getResult());
            }
            std::reverse(concatInputs.begin(), concatInputs.end());

            auto concatOp = rewriter.create<IE::ConcatOp>(takeOpLoc(origOp, "concat_in"), concatInputs, Dim(axis));
            source = concatOp.getOutput();
        }
        auto sizes = std::vector<int64_t>(plan.ends.size());
        std::transform(plan.ends.cbegin(), plan.ends.cend(), plan.begins.cbegin(), sizes.begin(),
                       std::minus<int64_t>());

        auto it = std::adjacent_find(sizes.begin(), sizes.end());
        auto newInputShape = std::vector<int64_t>(getShape(source).raw());
        auto index = it - sizes.begin();
        // Merge the adjacent dims if their values are both 1 and not slice dim
        // For example: 1x1x9x16x1000 -> 1x1x9x16x478
        // Only handle cases with input tensor rank greater than 4 and convert it to 4D tensor
        if (it != sizes.end() && *it == 1 && newInputShape[index] == 1 && newInputShape.size() > 4) {
            sizes.erase(it);
            auto newBegin = std::vector<int64_t>(plan.begins);
            newBegin.erase(newBegin.begin() + index);
            beginAttr = getIntArrayAttr(getContext(), newBegin);
            newInputShape.erase(newInputShape.begin() + index);
            source = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(
                    rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, StringLiteral("slice_{0}"), index), source,
                                                         nullptr, false, getIntArrayAttr(getContext(), newInputShape)));
        }
        const auto endsAttr = getIntArrayAttr(getContext(), sizes);
        newOp = rewriter.create<IE::SliceOp>(takeOpLoc(origOp, "slice_source"), source, beginAttr, endsAttr);
    }

    auto outputShape = plan.reshape_out_shape;
    if (outputShape.empty()) {
        outputShape.push_back(1);
    }
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);
    auto outReshape =
            rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, newOp->getResult(0), nullptr, false, outputShapeAttr);
    extendOpLoc(outReshape, "reshape_out");

    _log.trace("Replaced with 'IE::StridedSlice' -> 'IE::Reshape'");

    return mlir::success();
}

//
// safeRunOnFunc
//

void ResolveStridedSlicePass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isLegalOp = [&](IE::StridedSliceOp slice) {
        if (!slice.getBeginsAttr().has_value() || !slice.getEndsAttr().has_value() ||
            !slice.getStridesAttr().has_value()) {
            return true;
        }

        // Do not convert dynamic strided slice to IE.Slice with IE.Reshape.
        // Dynamic strided slice must be converted to an activation shave layer.
        if (getShape(slice.getInput()).isDynamic()) {
            return true;
        }

        auto isOne = [](auto val) {
            return val == 1;
        };

        VPUX_THROW_UNLESS(slice.getBeginsAttr().has_value(), "begins_attr is null");
        VPUX_THROW_UNLESS(slice.getEndsAttr().has_value(), "ends_attr is null");
        VPUX_THROW_UNLESS(slice.getStridesAttr().has_value(), "strides_attr is null");

        auto inputRank = getShape(slice.getInput()).size();
        auto beginsRank = slice.getBeginsAttr().value().size();
        auto endsRank = slice.getEndsAttr().value().size();
        auto stridesRank = slice.getStridesAttr().value().size();

        bool hasSameRank = (beginsRank == inputRank) && (endsRank == inputRank) && (stridesRank == inputRank);

        return slice.isSimplified() &&
               !llvm::all_of(parseIntArrayAttr<int64_t>(slice.getStridesAttr().value()), isOne) && hasSameRank;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::StridedSliceOp>(isLegalOp);
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::SliceOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SlicePlanning>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createResolveStridedSlicePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createResolveStridedSlicePass(Logger log) {
    return std::make_unique<ResolveStridedSlicePass>(log);
}
