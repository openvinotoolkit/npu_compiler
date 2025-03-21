//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTLARGECONVTOMULTICONVWITHADD
#define GEN_PASS_DEF_CONVERTLARGECONVTOMULTICONVWITHADD
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ConvertLargeConvToMultiConvWithAddPass
//

class ConvertLargeConvToMultiConvWithAddPass final :
        public IE::impl::ConvertLargeConvToMultiConvWithAddBase<ConvertLargeConvToMultiConvWithAddPass> {
public:
    explicit ConvertLargeConvToMultiConvWithAddPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class SplitConvToMultiConvWithAddConverter;

private:
    void safeRunOnFunc() final;
};

//
// SplitConvToMultiConvWithAddConverter
//

class ConvertLargeConvToMultiConvWithAddPass::SplitConvToMultiConvWithAddConverter final :
        public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    SplitConvToMultiConvWithAddConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Example: IC / OC = 2
//
//     Input (1xICxHxW)         Filter (OCxICxKYxKX)
//                    \         /
//                    Convolution (1xOCxHxW)
//                         |
//                      Output (1xOCxHxW)
//
// Convert to:
//
//     Input0 (1xIC/2xHxW)  Filter0 (OCxIC/2xKYxKX)   Input1 (1xIC/2xHxW)  Filter1 (OCxIC/2xKYxKX)
//                    \         /                                    \         /
//                    Convolution0 (1xOCxHxW)               Convolution1 (1xOCxHxW)
//                                          \               /
//                                            Add (1xOCxHxW)
//                                                   |
//                                           Output (1xOCxHxW)

mlir::LogicalResult ConvertLargeConvToMultiConvWithAddPass::SplitConvToMultiConvWithAddConverter::matchAndRewrite(
        IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    auto inputType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    auto filterType = mlir::cast<NDTypeInterface>(origOp.getFilter().getType());
    if (inputType.getRank() != 4 || filterType.getRank() != 4) {
        return mlir::failure();
    }

    auto filterShape = filterType.getShape();
    auto inputShape = inputType.getShape();
    const auto inChannels = filterShape[Dims4D::Filter::IC];
    const auto outChannels = filterShape[Dims4D::Filter::OC];
    const auto kernelWidth = filterShape[Dims4D::Filter::KX];
    const auto kernelHeight = filterShape[Dims4D::Filter::KY];

    auto module = origOp.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto numClusters = IE::getTileExecutor(module).getCount();
    const auto availableCMXSizePerCluster = vpux::VPU::getTotalCMXSize(origOp).count();
    const auto totalAvailableCMXSize = availableCMXSizePerCluster * numClusters;

    auto getOperandAllocSize = [&](mlir::Value operand) {
        auto operandType = mlir::cast<NDTypeInterface>(operand.getType());

        if (auto fakeQuantizeOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(operand.getDefiningOp())) {
            auto outLowConst = fakeQuantizeOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
            auto outHighConst = fakeQuantizeOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();
            if (outLowConst != nullptr && outHighConst != nullptr) {
                const auto realElemType = operandType.getElementType().cast<mlir::FloatType>();
                const auto operandQuantType = getQuantizedType(
                        outLowConst.getContentAttr(), outHighConst.getContentAttr(), fakeQuantizeOp.getLevels(),
                        fakeQuantizeOp.getLowFpType(), realElemType, /*isSigned=*/false, fakeQuantizeOp.getLoc(),
                        fakeQuantizeOp.getAutoBroadcast(), /*ignoreZPCheck=*/false, _log);
                if (operandQuantType != nullptr) {
                    return operandType.changeElemType(operandQuantType).getTotalAllocSize().count();
                }
            }
        }

        return operandType.getTotalAllocSize().count();
    };

    auto inputAllocSize = getOperandAllocSize(origOp.getInput());
    auto filterAllocSize = getOperandAllocSize(origOp.getFilter());

    // Experiment criteria for deciding whether to convert the convolution:
    // 1. Input and filter sizes are large enough to result in a large number of tiles
    //    even when tiling in two dimensions
    // 2. Kernels larger than 3 result in overlapping data
    // Performance benefits of this conversion:
    // 1. Reduces overlapping input data, thereby decreasing the DMA size
    // 2. Reduces the number of tiles, preventing excessive tiling and improving workload efficiency

    int minAllocSize = std::min(inputAllocSize, filterAllocSize);
    int maxAllocSize = std::max(inputAllocSize, filterAllocSize);
    bool isBeneficialShapeSize = (minAllocSize > totalAvailableCMXSize) || (minAllocSize > availableCMXSizePerCluster &&
                                                                            maxAllocSize > 2 * totalAvailableCMXSize);
    bool isBeneficialFilterSize =
            (inChannels > outChannels) && (inChannels % outChannels == 0) && (kernelWidth >= 3 && kernelHeight >= 3);

    if (!isBeneficialShapeSize || !isBeneficialFilterSize) {
        return mlir::failure();
    }

    _log.trace("Got target Convolution Op at '{0}' can be converted", origOp->getLoc());

    auto createSliceOp = [&](mlir::Value input, ShapeRef outShape, ShapeRef offsets, StringRef suffix) -> mlir::Value {
        return rewriter.create<IE::SliceOp>(takeOpLoc(origOp, suffix), input, offsets, outShape).getResult();
    };

    auto cloneOpAndReplaceInputs = [&](mlir::Operation* op, SmallVector<mlir::Value> origInputs,
                                       SmallVector<mlir::Value> newInputs, StringRef suffix) -> mlir::Value {
        mlir::IRMapping mapper;
        mapper.map(origInputs, newInputs);
        auto newOp = rewriter.clone(*op, mapper);
        vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ALL);

        newOp->setLoc(takeOpLoc(op, suffix));
        return newOp->getResult(0);
    };

    auto createAddOp = [&](mlir::Value input1, mlir::Value input2, StringRef suffix) -> mlir::Value {
        const auto broadcastType =
                vpux::IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);
        return rewriter
                .create<IE::AddOp>(takeOpLoc(origOp, suffix), input1, input2, broadcastType, nullptr, nullptr, nullptr,
                                   nullptr)
                .getResult();
    };

    auto inFq = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(origOp.getInput().getDefiningOp());
    auto filterFq = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(origOp.getFilter().getDefiningOp());

    const auto numSplits = inChannels / outChannels;
    int64_t channelOffset = 0;
    SmallVector<mlir::Value> convSlices;

    for (int64_t idx = 0; idx < numSplits; ++idx) {
        auto sliceInputShape =
                Shape{inputShape[Dims4D::Act::N], outChannels, inputShape[Dims4D::Act::H], inputShape[Dims4D::Act::W]};
        auto sliceOffset = Shape{0, channelOffset, 0, 0};
        channelOffset += outChannels;
        auto origInput = (inFq != nullptr) ? inFq.getInput() : origOp.getInput();
        auto newInput = createSliceOp(origInput, sliceInputShape, sliceOffset, "input_slice_" + std::to_string(idx));
        if (inFq != nullptr) {
            newInput = cloneOpAndReplaceInputs(inFq, {inFq.getInput()}, {newInput},
                                               "input_fq_slice_" + std::to_string(idx));
        }

        auto sliceFilterShape = Shape{outChannels, outChannels, kernelHeight, kernelWidth};
        auto origFilter = (filterFq != nullptr) ? filterFq.getInput() : origOp.getFilter();
        auto newFilter =
                createSliceOp(origFilter, sliceFilterShape, sliceOffset, "filter_slice_" + std::to_string(idx));
        if (filterFq != nullptr) {
            newFilter = cloneOpAndReplaceInputs(filterFq, {filterFq.getInput()}, {newFilter},
                                                "filter_fq_slice_" + std::to_string(idx));
        }

        auto newConv = cloneOpAndReplaceInputs(origOp, {origOp.getInput(), origOp.getFilter()}, {newInput, newFilter},
                                               "conv_slice_" + std::to_string(idx));

        // The bias only needs to exist once, regardless of which Convolution it is applied to
        if (origOp.getBias() != nullptr && idx != numSplits - 1) {
            newConv.getDefiningOp()->eraseOperand(2);
        }

        convSlices.push_back(newConv);
    }

    VPUX_THROW_UNLESS(convSlices.size() >= 2, "Got unexpect slice number");

    mlir::Value result = convSlices[0];
    for (size_t idx = 1; idx < convSlices.size(); ++idx) {
        result = createAddOp(result, convSlices[idx], "add_slice_" + std::to_string(idx));
    }

    rewriter.replaceOp(origOp, result);

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertLargeConvToMultiConvWithAddPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SplitConvToMultiConvWithAddConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertLargeConvToMultiConvWithAddPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertLargeConvToMultiConvWithAddPass(Logger log) {
    return std::make_unique<ConvertLargeConvToMultiConvWithAddPass>(log);
}
