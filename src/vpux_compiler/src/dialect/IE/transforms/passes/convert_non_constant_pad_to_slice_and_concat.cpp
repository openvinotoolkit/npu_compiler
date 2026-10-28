//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/WalkPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTNONCONSTANTPADTOSLICEANDCONCAT
#define GEN_PASS_DEF_CONVERTNONCONSTANTPADTOSLICEANDCONCAT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool shouldConvertPadOp(IE::PadOp op, bool hasSEPtrsEnabled, Logger log) {
    const auto logCb = [&](const formatv_object_base& msg) {
        log.trace("{0}", msg.str());
    };

    // Skip this rewriter if Pad is dynamic.
    if (IE::hasDynamicTensors(op)) {
        return false;
    }

    // Skip this rewriter if Pad can be executed using SEP.
    if (hasSEPtrsEnabled) {
        auto seOp = mlir::dyn_cast<IE::SEOpInterface>(op.getOperation());
        if (seOp && seOp.isSupported(logCb)) {
            log.nest().trace("Pad Operation {0} can be executed using SEP", op);
            return false;
        }
    }

    // Skip this rewriter for modes other than EDGE/REFLECT.
    return op.getMode() == IE::PadMode::EDGE || op.getMode() == IE::PadMode::REFLECT;
}

//
// ConvertNonConstantPadToSliceAndConcatPass
//

class ConvertNonConstantPadToSliceAndConcatPass final :
        public IE::impl::ConvertNonConstantPadToSliceAndConcatBase<ConvertNonConstantPadToSliceAndConcatPass> {
public:
    explicit ConvertNonConstantPadToSliceAndConcatPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class NonConstantPadConverter;

private:
    void safeRunOnFunc() final;
};

//
// NonConstantPadConverter
//

class ConvertNonConstantPadToSliceAndConcatPass::NonConstantPadConverter final :
        public mlir::OpRewritePattern<IE::PadOp> {
public:
    NonConstantPadConverter(mlir::MLIRContext* ctx, bool hasSEPtrsEnabled, Logger log)
            : mlir::OpRewritePattern<IE::PadOp>(ctx), _hasSEPtrsEnabled(hasSEPtrsEnabled), _log(log) {
        setDebugName("NonConstantPadConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::PadOp PadOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _hasSEPtrsEnabled;
    Logger _log;
};

mlir::Value convertPerAxisForEdgePad(mlir::Value input, const int64_t padBegin, const int64_t padEnd, const Dim padAxis,
                                     mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::MLIRContext* ctx) {
    if (padEnd == 0 && padBegin == 0) {
        return input;
    }

    auto inputShape = getShape(input);
    auto sliceData = [&](int64_t offset, StringRef locSuffix, int64_t idx) {
        auto offsets = SmallVector<int64_t>(inputShape.size(), 0);
        auto sizes = SmallVector<int64_t>(inputShape.begin(), inputShape.end());
        offsets[padAxis.ind()] = offset;
        sizes[padAxis.ind()] = 1;
        return IE::SliceOp::create(rewriter, appendLoc(loc, "slice_{0}_{1}_{2}", offset, locSuffix, idx), input,
                                   getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, sizes))
                .getResult();
    };

    SmallVector<mlir::Value> subSlices;
    for (auto i = 0; i < padBegin; ++i) {
        subSlices.push_back(sliceData(0, "edge_forward", int64_t(i)));
    }

    subSlices.push_back(input);

    for (auto i = 0; i < padEnd; ++i) {
        subSlices.push_back(sliceData(inputShape[padAxis] - 1, "edge_backward", int64_t(i)));
    }

    return IE::ConcatOp::create(rewriter, loc, subSlices, padAxis).getOutput();
}

mlir::Value convertPerAxisForReflectPad(mlir::Value input, const int64_t padBegin, const int64_t padEnd,
                                        const Dim padAxis, mlir::PatternRewriter& rewriter, mlir::Location loc,
                                        mlir::MLIRContext* ctx) {
    if (padEnd == 0 && padBegin == 0) {
        return input;
    }

    auto inputShape = getShape(input);
    auto sliceData = [&](int64_t offset, StringRef locSuffix) {
        auto offsets = SmallVector<int64_t>(inputShape.size(), 0);
        auto sizes = SmallVector<int64_t>(inputShape.begin(), inputShape.end());
        offsets[padAxis.ind()] = offset;
        sizes[padAxis.ind()] = 1;
        return IE::SliceOp::create(rewriter, appendLoc(loc, "slice_{0}_{1}", offset, locSuffix), input,
                                   getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, sizes))
                .getResult();
    };

    SmallVector<mlir::Value> subSlices;
    for (auto idx = padBegin; idx > 0; --idx) {
        const auto offset = idx;
        subSlices.push_back(sliceData(offset, "reflect_forward"));
    }

    subSlices.push_back(input);

    for (auto idx = 1; idx <= padEnd; ++idx) {
        const auto offset = inputShape[padAxis] - 1 - idx;
        subSlices.push_back(sliceData(offset, "reflect_backward"));
    }

    return IE::ConcatOp::create(rewriter, loc, subSlices, padAxis).getOutput();
}

mlir::LogicalResult ConvertNonConstantPadToSliceAndConcatPass::NonConstantPadConverter::matchAndRewrite(
        IE::PadOp padOp, mlir::PatternRewriter& rewriter) const {
    if (!shouldConvertPadOp(padOp, _hasSEPtrsEnabled, _log)) {
        return mlir::failure();
    }

    _log.trace("[{0}] Found '{1}' at '{2}'", getDebugName(), padOp->getName(), padOp->getLoc());

    VPUX_THROW_UNLESS(padOp.getPadsBeginAttr().has_value() && padOp.getPadsEndAttr().has_value(),
                      "Cannot get pad begin and pad end value");
    const auto padsBegin = parseIntArrayAttr<int64_t>(padOp.getPadsBeginAttr().value());
    const auto padsEnd = parseIntArrayAttr<int64_t>(padOp.getPadsEndAttr().value());

    mlir::Value input = padOp.getInput();
    auto inputRank = mlir::cast<mlir::RankedTensorType>(input.getType()).getRank();
    for (auto axis = inputRank - 1; axis >= 0; --axis) {
        auto newLoc = appendLoc(padOp.getLoc(), "pad_axis_{0}", Dim(axis));
        if (padOp.getMode() == IE::PadMode::REFLECT) {
            input = convertPerAxisForReflectPad(input, padsBegin[axis], padsEnd[axis], Dim(axis), rewriter, newLoc,
                                                padOp.getContext());
        }
        if (padOp.getMode() == IE::PadMode::EDGE) {
            input = convertPerAxisForEdgePad(input, padsBegin[axis], padsEnd[axis], Dim(axis), rewriter, newLoc,
                                             padOp.getContext());
        }
    }

    rewriter.replaceOp(padOp, input);

    _log.trace("Accomplished conversion of non constant pad to slice and concat");

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertNonConstantPadToSliceAndConcatPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto moduleOp = getModuleOp(func);
    const auto hasSEPtrsEnabled = config::hasEnableSEPtrsOperations(moduleOp);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<NonConstantPadConverter>(&ctx, hasSEPtrsEnabled, _log);

    walkAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createConvertNonConstantPadToSliceAndConcatPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertNonConstantPadToSliceAndConcatPass(Logger log) {
    return std::make_unique<ConvertNonConstantPadToSliceAndConcatPass>(log);
}
