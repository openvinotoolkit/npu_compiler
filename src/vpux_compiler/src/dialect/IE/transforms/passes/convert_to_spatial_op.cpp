//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/roll_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/se_roll_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTTOSPATIALOP
#define GEN_PASS_DEF_CONVERTTOSPATIALOP
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

mlir::Value createInputTranspose(mlir::AffineMap inAffineMap, mlir::Value input, mlir::Location loc,
                                 mlir::PatternRewriter& rewriter, Logger log) {
    auto inOrderAttr = mlir::AffineMapAttr::get(inAffineMap);
    log.trace("Create input Transpose with order attribute {0}", inOrderAttr);
    return rewriter.create<IE::TransposeOp>(loc, input, nullptr, inOrderAttr).getOutput();
}

mlir::Value createOutTranspose(mlir::AffineMap inAffineMap, mlir::Value input, mlir::Location loc,
                               mlir::PatternRewriter& rewriter, Logger log) {
    auto outOrderMap = mlir::inversePermutation(inAffineMap);
    auto outOrderAttr = mlir::AffineMapAttr::get(outOrderMap);
    log.trace("Create output Transpose with order attribute {0}", outOrderAttr);
    return rewriter.create<IE::TransposeOp>(loc, input, nullptr, outOrderAttr).getOutput();
}

//
// TransposeInterpolation
//

class TransposeInterpolation final : public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    TransposeInterpolation(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::InterpolateOp>(ctx), _log(log) {
        this->setDebugName("TransposeInterpolation");
    }

    SmallVector<int64_t> getInterpolationAxes(IE::InterpolateOp op) const;
    bool isSpatialInterpolation(ArrayRef<int64_t> interpolationAxes, int64_t shapeRank) const;

private:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Get the real interpolation axes by IO shapes
// There are cases where the axis attribute contains all four dimensions but the interpolation is done only on the
// spatial axes, so only the IO shapes can define the interpolation.
SmallVector<int64_t> TransposeInterpolation::getInterpolationAxes(IE::InterpolateOp op) const {
    const auto inputShape = getShape(op.getInput());
    const auto outputShape = getShape(op.getOutput());
    const auto rank = inputShape.size();

    SmallVector<int64_t> interpolationAxes;
    for (size_t i = 0; i < checked_cast<size_t>(rank); i++) {
        if (inputShape[Dim(i)] != outputShape[Dim(i)]) {
            interpolationAxes.push_back(checked_cast<int64_t>(i));
        }
    }

    // Adjust for single interpolation axis
    if (interpolationAxes.size() == 1) {
        if (interpolationAxes[0] == checked_cast<int64_t>(rank) - 1) {
            interpolationAxes.insert(interpolationAxes.begin(), interpolationAxes[0] - 1);
        } else {
            interpolationAxes.push_back(interpolationAxes[0] + 1);
        }
    }

    return interpolationAxes;
}

bool TransposeInterpolation::isSpatialInterpolation(ArrayRef<int64_t> interpolationAxes, int64_t shapeRank) const {
    for (size_t i = 0; i < interpolationAxes.size(); i++) {
        if (interpolationAxes[interpolationAxes.size() - 1 - i] != shapeRank - 1 - checked_cast<int64_t>(i)) {
            return false;
        }
    }

    return true;
}

mlir::LogicalResult TransposeInterpolation::matchAndRewrite(IE::InterpolateOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    if (!origOp.getSizesAttr().has_value() || !origOp.getScalesAttr().has_value() ||
        !origOp.getAxesAttr().has_value()) {
        _log.trace("InterpolateOp {0} at {1} has no size scale or axes attribute", origOp->getName(), origOp->getLoc());
        return mlir::failure();
    }

    const auto realInterpolationAxes = getInterpolationAxes(origOp);
    const auto inputShape = getShape(origOp.getInput());
    const auto rank = inputShape.size();

    if (isSpatialInterpolation(realInterpolationAxes, rank)) {
        _log.trace("Bypass {0} at {1}, which is spatial interpolaton already", origOp->getName(), origOp->getLoc());
        return mlir::failure();
    }

    auto ctx = rewriter.getContext();

    _log.trace("Got non-spatial interpolation {0} at {1}, interpolation axes {2}", origOp->getName(), origOp->getLoc(),
               realInterpolationAxes);

    // Create input Transpose
    SmallVector<uint32_t> inMemPerm;
    for (size_t i = 0; i < rank; i++) {
        if (std::find(realInterpolationAxes.begin(), realInterpolationAxes.end(), i) == realInterpolationAxes.end()) {
            inMemPerm.push_back(checked_cast<uint32_t>(i));
        }
    }
    for (const auto& axis : realInterpolationAxes) {
        inMemPerm.push_back(checked_cast<uint32_t>(axis));
    }

    const auto inOrderMap = mlir::AffineMap::getPermutationMap(inMemPerm, ctx);
    auto inTranspose =
            createInputTranspose(inOrderMap, origOp.getInput(), takeOpLoc(origOp, "transpose_in"), rewriter, _log);

    // Create new Interpolate
    auto origAxesValue = parseIntArrayAttr<int64_t>(origOp.getAxesAttrAttr());
    auto origScalesValue = parseFPArrayAttr<double>(origOp.getScalesAttrAttr());
    auto origSizesValue = parseIntArrayAttr<int64_t>(origOp.getSizesAttrAttr());

    SmallVector<int64_t> newAxesValue;
    SmallVector<double> newScalesValue;
    SmallVector<int64_t> newSizesValue;
    if (origAxesValue.size() == 2) {
        for (size_t i = 0; i < origAxesValue.size(); i++) {
            newAxesValue.insert(newAxesValue.begin(), checked_cast<int64_t>(rank - 1 - i));
        }
        newScalesValue.assign(origScalesValue);
        newSizesValue.assign(origSizesValue);
    } else if (origAxesValue.size() == rank) {
        newAxesValue.assign(origAxesValue);
        for (size_t i = 0; i < origScalesValue.size(); i++) {
            newScalesValue.push_back(origScalesValue[inMemPerm[i]]);
        }
        for (size_t i = 0; i < origSizesValue.size(); i++) {
            newSizesValue.push_back(origSizesValue[inMemPerm[i]]);
        }
    } else {
        return matchFailed(_log, rewriter, origOp, "InterpolateOp {0} has invalid axes attribute", origOp->getLoc());
    }

    const auto newAxesAttr = getIntArrayAttr(ctx, newAxesValue);
    const auto newScalesAttr = getFPArrayAttr(ctx, newScalesValue);
    const auto newSizesAttr = getIntArrayAttr(ctx, newSizesValue);
    _log.nest().trace("Create new Interpolate with axes attr {0}, scales attr {1}, sizes attr {2}", newAxesAttr,
                      newScalesAttr, newSizesAttr);
    auto newInterpolate = rewriter.create<IE::InterpolateOp>(
            origOp->getLoc(), inTranspose, origOp.getSizes(), origOp.getScales(), origOp.getAxes(), newSizesAttr,
            newScalesAttr, newAxesAttr, origOp.getTileOffsetAttrAttr(), origOp.getInitialInputDimsAttrAttr(),
            origOp.getInitialOutputDimsAttrAttr(), origOp.getAttr(), origOp.getOutputChannelsAttr());

    // Create output Transpose
    auto outTransposeOutput = createOutTranspose(inOrderMap, newInterpolate.getOutput(),
                                                 takeOpLoc(origOp, "transpose_out"), rewriter, _log);

    _log.trace("Finished replacement at {0}", origOp->getLoc());
    rewriter.replaceOp(origOp, outTransposeOutput);

    return mlir::success();
}

//
// TransposeRoll
//

class TransposeRoll final : public mlir::OpRewritePattern<IE::RollOp> {
public:
    TransposeRoll(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::RollOp>(ctx), _log(log) {
        setDebugName("TransposeRoll");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::RollOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TransposeRoll::matchAndRewrite(IE::RollOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp.getLoc());

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    if (VPU::isSupportedSEPRoll(origOp, logCb, /*checkLayout=*/false, /*checkChannelAlignment=*/true)) {
        return mlir::failure();
    }

    const auto inputType = origOp.getData().getType().cast<vpux::NDTypeInterface>();
    if (inputType.getRank() != 4) {
        _log.trace("only 4D input supported");
        return mlir::failure();
    }

    const auto inputShape = inputType.getShape();
    auto shiftAndAxesOrFail =
            IE::getShiftAndAxesForRollOp(origOp.getLoc(), origOp.getShift(), origOp.getAxes(), inputShape);
    if (mlir::failed(shiftAndAxesOrFail)) {
        return mlir::failure();
    }
    const auto shiftAndAxes = shiftAndAxesOrFail.value();
    const auto axes = shiftAndAxes.axes;
    if (axes.size() > 2) {
        _log.trace("only roll at 1D or 2D supported");
        return mlir::failure();
    }

    // Avoid introducing extra MemPermute, only 2 cases below are considered
    //    NxCxHxW@axes<1,2> -> NxWxCxH@axes<2,3>
    //    NxCxHxW@axes<1>   -> NxWxCxH@axes<2>
    //
    // since Transpose+Reorder(NCHW->NHWC) can be fused to PermuteCast
    //
    if (axes.size() == 1 && axes[0] != Dims4D::Act::C.ind()) {
        return mlir::failure();
    }
    if (axes.size() == 2 && (axes[0] != Dims4D::Act::C.ind() || axes[1] != Dims4D::Act::H.ind())) {
        return mlir::failure();
    }

    const auto newInAffineMap =
            getPermutationFromOrders(inputType.getDimsOrder(), DimsOrder::NWCH, rewriter.getContext());
    const auto newInOrder = DimsOrder::fromAffineMap(newInAffineMap);
    const auto newInShapeAfterTranspose = newInOrder.toMemoryOrder(inputShape);

    const auto align = vpux::VPU::NCEInvariant::getAlignment(inputType.getElementType());
    if (newInShapeAfterTranspose[MemDim(Dims4D::Act::N.ind())] != 1 ||
        newInShapeAfterTranspose[MemDim(Dims4D::Act::C.ind())] % align != 0) {
        _log.trace("Cannot convert to SE Roll after transpose ");
        return mlir::failure();
    }

    const auto newRollInput =
            createInputTranspose(newInAffineMap, origOp.getData(), takeOpLoc(origOp, "transpose_in"), rewriter, _log);

    const auto newAxes = axes.size() == 1 ? SmallVector<int32_t>{Dims4D::Act::H.ind()}
                                          : SmallVector<int32_t>{Dims4D::Act::H.ind(), Dims4D::Act::W.ind()};
    const auto newAxesElems = checked_cast<int64_t>(axes.size());
    const auto axesDimOrder = origOp.getAxes().getType().cast<vpux::NDTypeInterface>().getDimsOrder();
    const auto newAxesType =
            mlir::RankedTensorType::get(ArrayRef(newAxesElems), origOp.getAxes().getType().getElementType(),
                                        getTensorAttr(rewriter.getContext(), axesDimOrder, nullptr, nullptr));
    const auto newAxesValue = Const::createConst(rewriter, origOp.getAxes().getLoc(), newAxesType, ArrayRef(newAxes));

    auto newRollOp = rewriter.create<IE::RollOp>(origOp.getLoc(), newRollInput.getType(), newRollInput,
                                                 origOp.getShift(), newAxesValue);

    _log.trace("new Roll {0}", newRollOp);

    const auto outTransposeOutput = createOutTranspose(newInAffineMap, newRollOp.getOutput(),
                                                       takeOpLoc(origOp, "transpose_out"), rewriter, _log);
    rewriter.replaceOp(origOp, outTransposeOutput);

    return mlir::success();
}

//
// ConvertToSpatialOpPass
//

class ConvertToSpatialOpPass final : public IE::impl::ConvertToSpatialOpBase<ConvertToSpatialOpPass> {
public:
    explicit ConvertToSpatialOpPass(const bool m2iEnabled, const bool seExperimentalOpsEnabled, Logger log)
            : _m2iEnabled(m2iEnabled), _seExperimentalOpsEnabled(seExperimentalOpsEnabled), _log(log) {
        _log.setName(Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    bool _m2iEnabled;
    bool _seExperimentalOpsEnabled;
    Logger _log;
};

mlir::LogicalResult ConvertToSpatialOpPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (m2iEnabled.hasValue()) {
        _m2iEnabled = m2iEnabled.getValue();
    }

    if (seExperimentalOpsEnabled.hasValue()) {
        _seExperimentalOpsEnabled = seExperimentalOpsEnabled.getValue();
    }

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertToSpatialOpPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    if (!_m2iEnabled) {
        patterns.add<TransposeInterpolation>(&ctx, _log);
        IE::InterpolateOp::getCanonicalizationPatterns(patterns, &ctx);
    }

    if (_seExperimentalOpsEnabled) {
        patterns.add<TransposeRoll>(&ctx, _log);
    }

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertToSpatialOpPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertToSpatialOpPass(const bool m2iEnabled,
                                                                   const bool seExperimentalOpsEnabled, Logger log) {
    return std::make_unique<ConvertToSpatialOpPass>(m2iEnabled, seExperimentalOpsEnabled, log);
}
