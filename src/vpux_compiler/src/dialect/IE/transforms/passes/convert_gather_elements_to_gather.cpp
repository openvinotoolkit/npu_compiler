//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTGATHERELEMENTSTOGATHER
#define GEN_PASS_DEF_CONVERTGATHERELEMENTSTOGATHER
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
//   Convert GatherElementsOp to GatherOp
//
//   input0: 1x1x5376x80    input1: 1x1x300x1
//        |                     |
//        |               ┌────────────┐
//        │               │    Tile    │ repeats_values: [1, 1, 1, 80]
//        │               └─────┬──────┘
//         \                   / 1x1x300x80
//           ┌────────────────┐
//           │ GatherElements │ axis = 2
//           └────────┬───────┘
//                    │
//           output: 1x1x300x80
//
//                 ======►
//
//   input0: 1x1x5376x80    input1: 1x1x300x1
//        |                     |
//        |               ┌────────────┐
//        │               │   Squeeze  │ axes_values: [0, 1, 3]
//        │               └─────┬──────┘
//         \                   / 300
//           ┌────────────────┐
//           │     Gather     │ axis_value = 2
//           └────────┬───────┘
//                    │
//           output: 1x1x300x80
//

//
// GatherElementsOpConverter
//

class GatherElementsOpConverter final : public mlir::OpRewritePattern<IE::GatherElementsOp> {
public:
    GatherElementsOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GatherElementsOp>(ctx), _log(std::move(log)) {
        setDebugName("GatherElementsOpConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GatherElementsOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GatherElementsOpConverter::matchAndRewrite(IE::GatherElementsOp origOp,
                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto& ctx = origOp.getContext();
    const mlir::Location location = origOp.getLoc();

    auto maybeTileOp = mlir::dyn_cast_or_null<IE::TileOp>(origOp.getOperand(1).getDefiningOp());
    if (maybeTileOp == nullptr) {
        return matchFailed(rewriter, origOp, "No parent TileOp found");
    }

    auto tileOpInShape = getShape(maybeTileOp.getInput());
    SmallVector<Dim> nonOneDims = getNonOneDim(tileOpInShape);
    if (nonOneDims.size() > 1) {
        return matchFailed(rewriter, maybeTileOp, "Not supported TileOp with input shape of non-one dims size > 1");
    }

    // Get repeatsValues from TileOp
    auto repeatsValues = maybeTileOp.getRepeatsValues();
    if (!repeatsValues.has_value()) {
        return matchFailed(rewriter, maybeTileOp, "No repeats values found");
    }

    auto getNonOneRepeatAxesValue = [](mlir::ArrayAttr repeatsValues) -> SmallVector<std::pair<size_t, size_t>> {
        SmallVector<std::pair<size_t, size_t>> repeatAxesValue;
        auto repeatsVector = parseIntArrayAttr<int64_t>(repeatsValues);
        for (const auto& repeatValue : repeatsVector | indexed) {
            if (repeatValue.value() != 1) {
                repeatAxesValue.emplace_back(repeatValue.index(), repeatValue.value());
            }
        }
        return repeatAxesValue;
    };

    auto repeatAxesValue = getNonOneRepeatAxesValue(repeatsValues.value());
    if (repeatAxesValue.size() != 1) {
        return matchFailed(rewriter, maybeTileOp, "Not one repeat axis");
    }

    int64_t repeatAxis = repeatAxesValue[0].first;
    if (nonOneDims[0] == Dim(repeatAxis)) {
        return matchFailed(rewriter, maybeTileOp, "Not supported repeat axis");
    }

    // Get axis from GatherElementsOp
    int64_t axis = origOp.getAxis();
    if (Dim(axis) != nonOneDims[0]) {
        return matchFailed(rewriter, origOp, "Not supported GatherElementsOp axis");
    }

    auto origOpInShape = getShape(origOp.getInput());
    if (origOpInShape[Dim(repeatAxis)] != (int64_t)repeatAxesValue[0].second) {
        return matchFailed(rewriter, origOp, "Not supported GatherElementsOp shape");
    }

    auto generateAxesValue = [](size_t shapeSize, int64_t nonOneAxis) -> SmallVector<size_t> {
        SmallVector<size_t> axisOneArray;
        for (auto index : irange(shapeSize)) {
            if (index != (size_t)nonOneAxis) {
                axisOneArray.push_back(index);
            }
        }
        return axisOneArray;
    };

    // Create SqueezeOp
    auto axisOneArray = generateAxesValue(tileOpInShape.size(), axis);
    const auto axisOneArrayAttr = getIntArrayAttr(ctx, axisOneArray);
    auto squeezeOp = rewriter.create<IE::SqueezeOp>(appendLoc(location, "_squeeze"), maybeTileOp.getInput(), nullptr,
                                                    axisOneArrayAttr);

    // Create GatherOp
    int64_t batchDims = 0;
    rewriter.replaceOpWithNewOp<IE::GatherOp>(origOp, origOp.getOperand(0), squeezeOp, nullptr, getIntAttr(ctx, axis),
                                              batchDims, nullptr);

    return mlir::success();
}

//
// ConvertGatherElementsToGatherPass
//

class ConvertGatherElementsToGatherPass final :
        public IE::impl::ConvertGatherElementsToGatherBase<ConvertGatherElementsToGatherPass> {
public:
    explicit ConvertGatherElementsToGatherPass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertGatherElementsToGatherPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GatherElementsOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertGatherElementsToGatherPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertGatherElementsToGatherPass(Logger log) {
    return std::make_unique<ConvertGatherElementsToGatherPass>(log);
}
