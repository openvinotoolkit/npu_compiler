//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_DILATEDCONVCONVERT
#define GEN_PASS_DEF_DILATEDCONVCONVERT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// DilatedConvConverter
//

template <class ConvType>
class DilatedConvConverter final : public mlir::OpRewritePattern<ConvType> {
public:
    DilatedConvConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConvType>(ctx), _log(log) {
        this->setDebugName("DilatedConvConverter");
    }

    mlir::LogicalResult matchAndRewrite(ConvType convOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConvType>
bool isLegalConvAttr(ConvType convOp, Logger log) {
    auto isAllEqualOneValue = [](const SmallVector<int64_t>& shape, const int64_t value) {
        return std::all_of(shape.begin(), shape.end(), [&](auto size) {
            return size == value;
        });
    };

    auto inputType = convOp.getInput().getType().template cast<NDTypeInterface>();
    if (inputType.getRank() != 4) {
        log.trace("'{0}' input rank should equal 4, but got '{1}'", convOp->getName(), inputType.getRank());
        return false;
    }

    if (!convOp.getResult().hasOneUse()) {
        log.trace("'{0}' should only have one user", convOp->getName());
        return false;
    }

    auto origDilations = parseIntArrayAttr<int64_t>(convOp.getDilationsAttr());
    if (!isAllEqualOneValue(origDilations, 1)) {
        log.trace("'{0}' dilations '{1}' should all equal one", convOp->getName(), origDilations);
        return false;
    }

    auto origStrides = parseIntArrayAttr<int64_t>(convOp.getStrides());
    if (!isAllEqualOneValue(origStrides, 1)) {
        log.trace("'{0}' strides '{1}' should all equal one", convOp->getName(), origStrides);
        return false;
    }

    auto origPadsBegin = parseIntArrayAttr<int64_t>(convOp.getPadsBegin());
    auto origPadsEnd = parseIntArrayAttr<int64_t>(convOp.getPadsEnd());
    if (!isAllEqualOneValue(origPadsBegin, 0) || !isAllEqualOneValue(origPadsEnd, 0)) {
        log.trace("'{0}' PadsBegin '{1}' and PadsEnd '{2}' should all equal zero", convOp->getName(), origPadsBegin,
                  origPadsEnd);
        return false;
    }

    return true;
}

bool isLegalSpaceBatchAttr(IE::SpaceToBatch spaceToBatchOp, IE::BatchToSpace batchToSpaceOp, Logger log) {
    if (!spaceToBatchOp.getBlockShapeValue().has_value() || !batchToSpaceOp.getBlockShapeValue().has_value()) {
        log.trace("Cannot get block shape");
        return false;
    }
    auto spaceToBatchBlockSize = parseIntArrayAttr<int64_t>(spaceToBatchOp.getBlockShapeValue().value());
    auto batchTospaceBlockSize = parseIntArrayAttr<int64_t>(batchToSpaceOp.getBlockShapeValue().value());
    if (spaceToBatchBlockSize != batchTospaceBlockSize && spaceToBatchBlockSize[Dims4D::Act::N.ind()] == 1 &&
        spaceToBatchBlockSize[Dims4D::Act::C.ind()] == 1) {
        log.trace("SpaceToBatch block size '{0}' should equal with BatchToSpace block size '{1}' and equal one at N, C",
                  spaceToBatchBlockSize, batchTospaceBlockSize);
        return false;
    }

    if (!spaceToBatchOp.getPadsBeginValue().has_value() || !spaceToBatchOp.getPadsEndValue().has_value()) {
        log.trace("Cannot get pads value of spaceToBatchOp");
        return false;
    }

    if (!batchToSpaceOp.getCropsBeginValue().has_value() || !batchToSpaceOp.getCropsEndValue().has_value()) {
        log.trace("Cannot get crops value of batchToSpaceOp");
        return false;
    }

    auto spaceToBatchPadsBegin = parseIntArrayAttr<int64_t>(spaceToBatchOp.getPadsBeginValue().value());
    auto spaceToBatchPadsEnd = parseIntArrayAttr<int64_t>(spaceToBatchOp.getPadsEndValue().value());
    auto batchToSpaceCropsBegin = parseIntArrayAttr<int64_t>(batchToSpaceOp.getCropsBeginValue().value());
    auto batchToSpaceCropsEnd = parseIntArrayAttr<int64_t>(batchToSpaceOp.getCropsEndValue().value());

    // Pads and Crops should equal 0 at N and C
    // Pad value should larger or equal Crop value at H and W
    auto isLegalPadAndCropValue = [](SmallVector<int64_t>& padsValue, SmallVector<int64_t>& cropsValue) {
        for (size_t idx = 0; idx < padsValue.size(); idx++) {
            if (idx <= checked_cast<size_t>(Dims4D::Act::C.ind()) && (padsValue[idx] != 0 || cropsValue[idx] != 0)) {
                return false;
            }

            if (padsValue[idx] < cropsValue[idx]) {
                return false;
            }
        }
        return true;
    };

    if (!isLegalPadAndCropValue(spaceToBatchPadsBegin, batchToSpaceCropsBegin) ||
        !isLegalPadAndCropValue(spaceToBatchPadsEnd, batchToSpaceCropsEnd)) {
        log.trace("Illegal Pad value and Crop value");
        return false;
    }

    return true;
}

mlir::Value createDilatedConvOp(mlir::PatternRewriter& rewriter, IE::ConvolutionOp convOp, mlir::Value input,
                                mlir::ArrayAttr padsBeginAttr, mlir::ArrayAttr padsEndAttr,
                                mlir::ArrayAttr dilationsAttr) {
    return rewriter
            .create<IE::ConvolutionOp>(convOp.getLoc(), input, convOp.getFilter(), convOp.getBias(),
                                       convOp.getStridesAttr(), padsBeginAttr, padsEndAttr, dilationsAttr,
                                       convOp.getPostOpAttr(), convOp.getClampAttr(), convOp.getStaticScaleAttr(),
                                       convOp.getOutputChannelsAttr(), convOp.getInputChannelsAttr())
            .getResult();
}

mlir::Value createDilatedConvOp(mlir::PatternRewriter& rewriter, IE::GroupConvolutionOp groupConvOp, mlir::Value input,
                                mlir::ArrayAttr padsBeginAttr, mlir::ArrayAttr padsEndAttr,
                                mlir::ArrayAttr dilationsAttr) {
    return rewriter
            .create<IE::GroupConvolutionOp>(groupConvOp.getLoc(), input, groupConvOp.getFilter(), groupConvOp.getBias(),
                                            groupConvOp.getStridesAttr(), padsBeginAttr, padsEndAttr, dilationsAttr,
                                            groupConvOp.getGroupsAttr(), groupConvOp.getPostOpAttr(),
                                            groupConvOp.getClampAttr(), groupConvOp.getOutputChannelsAttr(),
                                            groupConvOp.getInputChannelsAttr())
            .getResult();
}

// Match pattern:
//      SpaceToBatch -> FakeQuantize(optional) -> Convolution / GroupConvolution -> BatchToSpace
// Convert to:
//      FakeQuantize(optional) -> Dilated Convolution / Dilated GroupConvolution
template <class ConvType>
mlir::LogicalResult DilatedConvConverter<ConvType>::matchAndRewrite(ConvType convOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op {0} at {1}", convOp->getName(), convOp->getLoc());

    auto batchToSpaceOp = mlir::dyn_cast<IE::BatchToSpace>(*convOp->getResult(0).getUsers().begin());
    if (batchToSpaceOp == nullptr) {
        _log.trace("Pattern mismatched: There is no BatchToSpace Op");
        return mlir::failure();
    }

    auto potentialFQOp = convOp.getInput().template getDefiningOp<IE::FakeQuantizeOp>();
    if (potentialFQOp != nullptr && !potentialFQOp.getResult().hasOneUse()) {
        _log.trace("Pattern mismatched: FakeQuantize Op should only have one user");
        return mlir::failure();
    }

    auto spaceToBatchOp = (potentialFQOp == nullptr)
                                  ? convOp.getInput().template getDefiningOp<IE::SpaceToBatch>()
                                  : potentialFQOp.getInput().template getDefiningOp<IE::SpaceToBatch>();
    if (spaceToBatchOp == nullptr || !spaceToBatchOp.getResult().hasOneUse()) {
        _log.trace("Pattern mismatched: There is no SpaceToBatch Op or with multi user");
        return mlir::failure();
    }

    if (!isLegalConvAttr(convOp, _log) || !isLegalSpaceBatchAttr(spaceToBatchOp, batchToSpaceOp, _log)) {
        return mlir::failure();
    }

    auto spaceToBatchBlockSize = parseIntArrayAttr<int64_t>(spaceToBatchOp.getBlockShapeValue().value());
    SmallVector<int64_t> newDilations(spaceToBatchBlockSize.begin() + 2, spaceToBatchBlockSize.end());

    auto spaceToBatchPadsBegin = parseIntArrayAttr<int64_t>(spaceToBatchOp.getPadsBeginValue().value());
    auto spaceToBatchPadsEnd = parseIntArrayAttr<int64_t>(spaceToBatchOp.getPadsEndValue().value());
    auto batchTospaceCropsBegin = parseIntArrayAttr<int64_t>(batchToSpaceOp.getCropsBeginValue().value());
    auto batchTospaceCropsEnd = parseIntArrayAttr<int64_t>(batchToSpaceOp.getCropsEndValue().value());
    auto getNewPadsValue = [](const SmallVector<int64_t>& pads, const SmallVector<int64_t>& crops) {
        SmallVector<int64_t> newPads;
        std::transform(pads.begin() + 2, pads.end(), crops.begin() + 2, std::back_inserter(newPads),
                       [](const int64_t pad, const int64_t crop) {
                           return pad - crop;
                       });
        return newPads;
    };
    auto newPadsBegin = getNewPadsValue(spaceToBatchPadsBegin, batchTospaceCropsBegin);
    auto newPadsEnd = getNewPadsValue(spaceToBatchPadsEnd, batchTospaceCropsEnd);

    rewriter.setInsertionPointAfter(batchToSpaceOp);
    auto newInput = spaceToBatchOp.getInput();
    if (potentialFQOp != nullptr) {
        potentialFQOp.setOperand(0, newInput);
        inferReturnTypes(potentialFQOp, vpux::InferShapedTypeMode::ALL);
        newInput = potentialFQOp.getResult();
    }

    const auto newPadsBeginAttr = getIntArrayAttr(spaceToBatchOp.getContext(), newPadsBegin);
    const auto newPadsEndAttr = getIntArrayAttr(spaceToBatchOp.getContext(), newPadsEnd);
    const auto newDilationsAttr = getIntArrayAttr(spaceToBatchOp.getContext(), newDilations);
    auto dilatedConvOp =
            createDilatedConvOp(rewriter, convOp, newInput, newPadsBeginAttr, newPadsEndAttr, newDilationsAttr);

    batchToSpaceOp.replaceAllUsesWith(dilatedConvOp);
    return mlir::success();
}

//
// DilatedConvConvertPass
//

class DilatedConvConvertPass final : public IE::impl::DilatedConvConvertBase<DilatedConvConvertPass> {
public:
    explicit DilatedConvConvertPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void DilatedConvConvertPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DilatedConvConverter<IE::ConvolutionOp>>(&ctx, _log);
    patterns.add<DilatedConvConverter<IE::GroupConvolutionOp>>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDilatedConvConvertPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDilatedConvConvertPass(Logger log) {
    return std::make_unique<DilatedConvConvertPass>(log);
}
