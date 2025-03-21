//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSERESHAPEMVN
#define GEN_PASS_DEF_FUSERESHAPEMVN
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Example of how MVN with 'internal_reshape' works in NHWC:
//
// Original-input(W=4)x(C=8) reshaped to (W=8)x(K=4) and back.
//        c0    c1    c2    c3    c4    c5    c6    c7                   K0    K1    K2    K3
// W=0 | 0x80  0x90  0xa0  0xb0  0xc0  0xd0  0xe0  0xf0           W=0 | 0x80  0xa0  0xc0  0xe0
// W=1 | 0x81  0x91  0xa1  0xb1  0xc1  0xd1  0xe1  0xf1           W=1 | 0x81  0xa1  0xc1  0xe1
// W=2 | 0x82  0x92  0xa2  0xb2  0xc2  0xd2  0xe2  0xf2 (reshape) W=2 | 0x82  0xa2  0xc2  0xe2
// W=3 | 0x83  0x93  0xa3  0xb3  0xc3  0xd3  0xe3  0xf3 =======>  W=3 | 0x83  0xa3  0xc3  0xe3
//                                                                W=4 | 0x90  0xb0  0xd0  0xf0
//                                                                W=5 | 0x91  0xb1  0xd1  0xf1
//                                                                W=6 | 0x92  0xb2  0xd2  0xf2
//                                                 ...  <=======  W=7 | 0x93  0xb3  0xd3  0xf3
//                                                      (reshape)

// MVN has to compute params & normalize K channels.
// But K0/K1/K2/K3 elements can be found in original-input, in nearby (C=8)/(K=4) = 2 channels.
// Example for K0:
//   K0 data is present in input {c0,c1} channels, so MVN with 'internal_reshape' will compute
//   norm/scale considering {c0,c1} are same channel and replicate correction params for {c0,c1} since they are
//   really the content of same K0 channel. This way, physical permutation of data in memory can be avoided.

//
// ReshapeMVNPattern
//

class ReshapeMVNPattern {
public:
    ReshapeMVNPattern(IE::MVNOp mvnOp, Logger log): _mvnOp(mvnOp), _log(log) {
    }

    bool init();

private:
    IE::MVNOp _mvnOp;
    mlir::Value _patternIn = nullptr;
    mlir::Value _patternOut = nullptr;
    IE::GroupConvolutionOp _groupConvOp = nullptr;
    int64_t _origChannelSize = 0;
    int64_t _newChannelSize = 0;

    Logger _log;

public:
    template <class TargetOp>
    TargetOp getTargetOpWithSpecificLayoutAndSingleUser(mlir::Operation* op, DimsOrder inTargetOrder,
                                                        DimsOrder outTargetOrder) const;

    bool isSupportedGroupConv();
    mlir::LogicalResult replacePattern();
};

template <class TargetOp>
TargetOp ReshapeMVNPattern::getTargetOpWithSpecificLayoutAndSingleUser(mlir::Operation* op, DimsOrder inTargetOrder,
                                                                       DimsOrder outTargetOrder) const {
    if (auto targetOp = mlir::dyn_cast_or_null<TargetOp>(op)) {
        if (targetOp->getResult(0).hasOneUse()) {
            auto inType = targetOp->getOperand(0).getType().template cast<NDTypeInterface>();
            auto outType = targetOp->getResult(0).getType().template cast<NDTypeInterface>();
            if (inType.getRank() == 4 && outType.getRank() == 4 && inType.getDimsOrder() == inTargetOrder &&
                outType.getDimsOrder() == outTargetOrder) {
                return targetOp;
            }
        }
    }
    return nullptr;
}

bool ReshapeMVNPattern::isSupportedGroupConv() {
    const auto log = _log.nest();

    if (_groupConvOp == nullptr) {
        log.trace("GroupConv match failed: No GroupConv operation found");
        return false;
    }

    const auto isGreaterThanOne = [](const int64_t val) {
        return val > 1;
    };

    const auto isNotEqualZero = [](const int64_t val) {
        return val != 0;
    };
    // Pattern matching in this pass is quite complex and slow, should be refactored to use
    // better options than applyPatternsAndFoldGreedily -> E#148655
    if (auto inFilter = _groupConvOp.getFilter(); !mlir::dyn_cast<Const::DeclareOp>(inFilter.getDefiningOp())) {
        _log.trace("GroupConvolution filter input not constant");
        // Filter may not be constant, it can be runtime dequantized and reordered
        // However this pattern will disable runtime quantization
        auto reorderOp = _groupConvOp.getFilter().getDefiningOp<IE::ReorderOp>();
        if (!reorderOp) {
            _log.trace("GroupConvolution filter !reorderOp");
            return false;
        }

        auto dequantize = reorderOp.getInput().getDefiningOp<IE::DequantizeOp>();
        if (!dequantize) {
            _log.trace("GroupConvolution filter !dequantize");
            return false;
        }
        auto quantizedConst = dequantize.getInput().getDefiningOp<Const::DeclareOp>();
        if (quantizedConst == nullptr) {
            _log.trace("GroupConvolution filter !quantizedConst");
            return false;
        }
    }

    if (_groupConvOp.getBias()) {
        log.trace("GroupConv match failed: GroupConv with bias is not implemented");
        return false;
    }

    auto filterShape = getShape(_groupConvOp.getFilter());
    if (filterShape[Dims4D::Filter::KX] != 1 || filterShape[Dims4D::Filter::KY] != 1 ||
        filterShape[Dims4D::Filter::OC] != _groupConvOp.getGroups().value() || filterShape[Dims4D::Filter::IC] != 1) {
        log.trace("GroupConv match failed: Unsupported filter shape");
        return false;
    }

    if (llvm::any_of(parseIntArrayAttr<int64_t>(_groupConvOp.getStrides()), isGreaterThanOne)) {
        log.trace("GroupConv match failed: Strides should all be one");
        return false;
    }
    if (llvm::any_of(parseIntArrayAttr<int64_t>(_groupConvOp.getDilations()), isGreaterThanOne)) {
        log.trace("GroupConv match failed: Dilations should all be one");
        return false;
    }
    if (llvm::any_of(parseIntArrayAttr<int64_t>(_groupConvOp.getPadsBegin()), isNotEqualZero)) {
        log.trace("GroupConv match failed: PadsBegin should all be zero");
        return false;
    }
    if (llvm::any_of(parseIntArrayAttr<int64_t>(_groupConvOp.getPadsEnd()), isNotEqualZero)) {
        log.trace("GroupConv match failed: PadsEnd should all be zero");
        return false;
    }

    return true;
}

// Base Pattern:
// From:
// Input(NHWC) -> Reorder(NCHW) -> Reshape(NCHW) -> Reorder(NHWC) -> MVN(NHWC) ->
//                Reorder(NCHW) -> Reshape(NCHW) -> Reorder(NHWC) -> Output(NHWC)
// To:
// Input(NHWC) -> MVN(NHWC) -> Output(NHWC)
//
// Variant 1:
// From:
// Input(NHWC) -> Reorder(NCHW) -> Reshape(NCHW) -> Reorder(NHWC) -> MVN(NHWC) -> Reorder(NCHW) ->
//         [AffineReshape(NCHW) -> Reorder(NHWC) -> GroupConv(NHWC) -> Reorder(NCHW)] ->
//                Reshape(NCHW) -> Reorder(NHWC) -> Output(NHWC)
// To:
// Input(NHWC) -> MVN(NHWC) -> GroupConv(NHWC) -> Output(NHWC)
//
// Variant 2:
// From:
// Concat(NCHW)
//   ├─ Reorder(NHWC) -> [...]
//   └─ Reshape(NCHW) -> Reorder(NHWC) -> MVN(NHWC) -> Reorder(NCHW) ->
//                       Reshape(NCHW) -> Reorder(NHWC) -> Output(NHWC)
// To:
// Concat(NCHW) -> Reorder(NHWC)
//                       ├─ [...]
//                       └─ MVN(NHWC) -> Output(NHWC)

bool ReshapeMVNPattern::init() {
    const auto log = _log.nest();
    const auto mvnInType = mlir::cast<NDTypeInterface>(_mvnOp.getInput().getType());

    if (mvnInType.getRank() != 4 || mvnInType.getDimsOrder() != DimsOrder::NHWC) {
        log.trace("Only support 4D MVN with NHWC layout but got {0}", mvnInType.getDimsOrder());
        return false;
    }

    if (!_mvnOp.getOutput().hasOneUse() || _mvnOp.getInternalReshape().has_value()) {
        log.trace("Only support single user MVN without Internal Reshape");
        return false;
    }

    // Check pattern before MVN Op:
    // Input(NHWC) -> Reorder1(NCHW) -> Reshape2(NCHW) -> Reorder3(NHWC) -> MVN(NHWC)
    auto reorder3 = getTargetOpWithSpecificLayoutAndSingleUser<IE::ReorderOp>(_mvnOp.getInput().getDefiningOp(),
                                                                              DimsOrder::NCHW, DimsOrder::NHWC);
    if (reorder3 == nullptr) {
        log.trace("Match failed: [Reorder]->MVN");
        return false;
    }
    auto reshape2 = getTargetOpWithSpecificLayoutAndSingleUser<IE::ReshapeOp>(reorder3.getInput().getDefiningOp(),
                                                                              DimsOrder::NCHW, DimsOrder::NCHW);
    if (reshape2 == nullptr) {
        log.trace("Match failed: [Reshape]->Reorder->MVN");
        return false;
    }
    auto reorder1 = getTargetOpWithSpecificLayoutAndSingleUser<IE::ReorderOp>(reshape2.getInput().getDefiningOp(),
                                                                              DimsOrder::NHWC, DimsOrder::NCHW);

    if (reorder1 != nullptr) {
        _patternIn = reorder1.getInput();
    } else {
        auto concatOp = mlir::dyn_cast_or_null<IE::ConcatOp>(reshape2.getInput().getDefiningOp());

        if (concatOp == nullptr) {
            log.trace("Match failed: [Reorder]->Reshape->Reorder->MVN");
            return false;
        }

        for (auto user : concatOp.getResult().getUsers()) {
            if (auto reorderInner = getTargetOpWithSpecificLayoutAndSingleUser<IE::ReorderOp>(user, DimsOrder::NCHW,
                                                                                              DimsOrder::NHWC)) {
                _patternIn = reorderInner.getOutput();
                break;
            }
        }
    }

    if (_patternIn == nullptr) {
        log.trace("Match failed: [Reorder]->Reshape->Reorder->MVN");
        return false;
    }

    // Check pattern after MVN Op:
    // MVN(NHWC) -> Reorder4(NCHW)
    auto reorder4 = getTargetOpWithSpecificLayoutAndSingleUser<IE::ReorderOp>(*(_mvnOp.getOutput().getUsers().begin()),
                                                                              DimsOrder::NHWC, DimsOrder::NCHW);
    if (reorder4 == nullptr) {
        log.trace("Match failed: MVN->[Reorder]");
        return false;
    }

    _patternOut = reorder4.getOutput();

    // Check [Optional Part] pattern
    // [AffineReshape(NCHW) -> Reorder(NHWC) -> GroupConv(NHWC) -> Reorder(NCHW)]
    if (auto affineReshape = getTargetOpWithSpecificLayoutAndSingleUser<IE::AffineReshapeOp>(
                *(_patternOut.getUsers().begin()), DimsOrder::NCHW, DimsOrder::NCHW)) {
        auto reorderPreGc = getTargetOpWithSpecificLayoutAndSingleUser<IE::ReorderOp>(
                *(affineReshape.getOutput().getUsers().begin()), DimsOrder::NCHW, DimsOrder::NHWC);
        if (reorderPreGc == nullptr) {
            log.trace("Match failed: MVN->AffineReshape->[Reorder]");
            return false;
        }

        _groupConvOp = getTargetOpWithSpecificLayoutAndSingleUser<IE::GroupConvolutionOp>(
                *(reorderPreGc.getOutput().getUsers().begin()), DimsOrder::NHWC, DimsOrder::NHWC);

        if (!isSupportedGroupConv()) {
            log.trace("Match failed: MVN->AffineReshape->Reorder->[GroupConv]");
            return false;
        }

        auto reorderPostGc = getTargetOpWithSpecificLayoutAndSingleUser<IE::ReorderOp>(
                *(_groupConvOp.getOutput().getUsers().begin()), DimsOrder::NHWC, DimsOrder::NCHW);
        if (reorderPostGc == nullptr) {
            log.trace("Match failed: MVN->AffineReshape->Reorder->GroupConv->[Reorder]");
            return false;
        }

        _patternOut = reorderPostGc.getOutput();
    }

    // Back to common pattern
    // MVN(NHWC) -> Reorder4(NCHW) -> [Optional Part](NCHW) -> Reshape5(NCHW) -> Reorder6(NHWC) -> Output(NHWC)
    auto reshape5 = getTargetOpWithSpecificLayoutAndSingleUser<IE::ReshapeOp>(*(_patternOut.getUsers().begin()),
                                                                              DimsOrder::NCHW, DimsOrder::NCHW);
    if (reshape5 == nullptr) {
        log.trace("Match failed: MVN->Reorder->(Optional Part)->[Reshape]");
        return false;
    }

    auto reorder6 = getTargetOpWithSpecificLayoutAndSingleUser<IE::ReorderOp>(
            *(reshape5.getOutput().getUsers().begin()), DimsOrder::NCHW, DimsOrder::NHWC);
    if (reorder6 == nullptr) {
        log.trace("Match failed: MVN->Reorder->(Optional Part)->Reshape->[Reorder]");
        return false;
    }

    _patternOut = reorder6.getOutput();

    // Check patern input and output has the same type
    auto patternInType = mlir::cast<NDTypeInterface>(_patternIn.getType());
    auto patternOutType = mlir::cast<NDTypeInterface>(_patternOut.getType());
    if (patternInType != patternOutType) {
        log.trace("Mismatching pattern input type {0} and output type {1}", patternInType, patternOutType);
        return false;
    }

    // Checks for C reshape value
    _newChannelSize = patternInType.getShape()[Dims4D::Act::C];
    _origChannelSize = mvnInType.getShape()[Dims4D::Act::C];
    if ((_newChannelSize % _origChannelSize) || (_newChannelSize <= _origChannelSize)) {
        log.trace("Expecting in-C to be a multiple of reshaped-C, got {0}, {1}", _newChannelSize, _origChannelSize);
        return false;
    }

    return true;
}

mlir::LogicalResult ReshapeMVNPattern::replacePattern() {
    mlir::OpBuilder builder(_mvnOp);
    auto ctx = builder.getContext();

    const auto mvnInputType = mlir::cast<NDTypeInterface>(_mvnOp.getInput().getType());
    auto internalReshapeAttr = getIntArrayAttr(ctx, mvnInputType.getShape());

    auto cloneOpAndReplaceInputs = [&](mlir::Operation* op, SmallVector<mlir::Value> origInputs,
                                       SmallVector<mlir::Value> newInputs) -> mlir::Operation* {
        mlir::IRMapping mapper;
        mapper.map(origInputs, newInputs);
        return builder.clone(*op, mapper);
    };

    auto newMvnOp = mlir::cast<IE::MVNOp>(cloneOpAndReplaceInputs(_mvnOp, {_mvnOp.getInput()}, {_patternIn}));
    newMvnOp.setInternalReshapeAttr(internalReshapeAttr);
    vpux::inferReturnTypes(newMvnOp, vpux::InferShapedTypeMode::ALL);

    auto patternOutVal = newMvnOp.getResult();
    if (_groupConvOp != nullptr) {
        auto filter = _groupConvOp.getFilter();
        auto filterConst = filter.getDefiningOp<Const::DeclareOp>();
        if (!filterConst) {
            auto reorderOp = filter.getDefiningOp<IE::ReorderOp>();
            auto permute = DimsOrder::fromAffineMap(reorderOp.getDstOrder());
            auto dequantize = reorderOp.getInput().getDefiningOp<IE::DequantizeOp>();
            auto quantizedConst = dequantize.getInput().getDefiningOp<Const::DeclareOp>();
            const auto qType = quantizedConst.getType().cast<vpux::NDTypeInterface>();
            const auto qElemType = qType.getElementType().cast<mlir::quant::QuantizedType>();
            const auto outType = dequantize.getType().cast<vpux::NDTypeInterface>();
            const auto newConstType = outType.changeElemType(qElemType.getExpressedType()).changeDimsOrder(permute);
            auto newConstAttr = quantizedConst.transformContentAttr().dequantize().reorder(permute).get();

            auto inFilter = builder.create<Const::DeclareOp>(filter.getLoc(), newConstType, std::move(newConstAttr));
            inFilter->setLoc(quantizedConst->getLoc());
            _log.trace("GroupConvolution filter dequantized");
            filterConst = inFilter;
        }
        const auto filterContent = filterConst.getContent();
        const auto filterVals = filterContent.getValues<float>();
        SmallVector<float> newFilterVals;
        if (filterContent.isSplat()) {
            newFilterVals.push_back(filterVals[0]);
        } else {
            int64_t repeatSize = _newChannelSize / _origChannelSize;
            for (int64_t idx = 0; idx < _origChannelSize; idx++) {
                newFilterVals.insert(newFilterVals.end(), repeatSize, filterVals[idx]);
            }
        }

        const auto newFilterShape = Shape{_newChannelSize, 1, 1, 1};
        const auto filterType = mlir::cast<NDTypeInterface>(filterConst.getOutput().getType());
        const auto dataStorageType = mlir::cast<mlir::RankedTensorType>(filterType.changeShape(newFilterShape));
        auto newFilter = Const::createFloatConst(builder, filter.getLoc(), dataStorageType, newFilterVals);

        auto newGroupConvOp = mlir::cast<IE::GroupConvolutionOp>(cloneOpAndReplaceInputs(
                _groupConvOp, {_groupConvOp.getInput(), _groupConvOp.getFilter()}, {newMvnOp.getResult(), newFilter}));
        newGroupConvOp.setGroupsAttr(getIntAttr(builder, _newChannelSize));
        vpux::inferReturnTypes(newGroupConvOp, vpux::InferShapedTypeMode::ALL);

        patternOutVal = newGroupConvOp.getResult();
    }
    _patternOut.replaceAllUsesWith(patternOutVal);

    _log.trace("Implementing fuse Reshape and MVN pattern");

    return mlir::success();
}

//
// FuseReshapeMvn
//

class FuseReshapeMvn final : public mlir::OpRewritePattern<IE::MVNOp> {
public:
    FuseReshapeMvn(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MVNOp>(ctx), _log(log) {
        setDebugName("FuseReshapeMvn");
    }

    mlir::LogicalResult matchAndRewrite(IE::MVNOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseReshapeMvn::matchAndRewrite(IE::MVNOp origOp, mlir::PatternRewriter& /*rewriter*/) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp.getLoc());
    auto pattern = ReshapeMVNPattern(origOp, _log);
    if (!pattern.init()) {
        return mlir::failure();
    }

    return pattern.replacePattern();
}

//
// FuseReshapeMvnPass
//

class FuseReshapeMvnPass final : public IE::impl::FuseReshapeMvnBase<FuseReshapeMvnPass> {
public:
    explicit FuseReshapeMvnPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void FuseReshapeMvnPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseReshapeMvn>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createFuseReshapeMvnPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseReshapeMvnPass(Logger log) {
    return std::make_unique<FuseReshapeMvnPass>(log);
}
