//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/expand_utils.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <numeric>

namespace vpux {
namespace IE {

//
// calcPadsEnd
//

Shape calcPadsEnd(ShapeRef origShape, ShapeRef extendedShape) {
    Shape padsEnd(origShape.size());

    for (auto i : irange(origShape.size())) {
        const auto d = Dim(i);
        padsEnd[d] = extendedShape[d] - origShape[d];
    }

    return padsEnd;
}

Shape calcPadsEnd(vpux::NDTypeInterface origType, int64_t channelAlignment) {
    const auto origShape = origType.getShape();

    auto extendedShape = origShape.toValues();
    extendedShape[Dims4D::Act::C] = alignValUp(origShape[Dims4D::Act::C], channelAlignment);

    return calcPadsEnd(origShape, extendedShape);
}

bool needsPadding(const int64_t dim) {
    return dim != 0;
}

SmallVector<int64_t> extractMeaningfulOutput(mlir::Operation* origOp, ShapeRef outPadsEnd) {
    SmallVector<int64_t> offsets(outPadsEnd.size(), 0);
    auto sliceOp = origOp->getOperand(0).template getDefiningOp<IE::SliceOp>();
    if (sliceOp != nullptr) {
        auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
        const auto sliceChannelOffset = sliceOffsets[Dims4D::Act::C.ind()];
        if (sliceChannelOffset < outPadsEnd[Dims4D::Act::C]) {
            offsets[Dims4D::Act::C.ind()] = sliceChannelOffset;
        } else {
            offsets[Dims4D::Act::C.ind()] = outPadsEnd[Dims4D::Act::C];
        }
    }

    return offsets;
}

mlir::Value expandWithOffset(mlir::PatternRewriter& rewriter, mlir::Operation* origOp, IE::SliceOp sliceOp,
                             mlir::Value expandValue, ShapeRef inPadsEnd, size_t expandDim) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(0).getType());
    const auto inputShape = inputType.getShape();

    auto padBegin = mlir::SmallVector<int64_t>(inputShape.size(), 0);
    auto padEnd = mlir::SmallVector<int64_t>(inputShape.size(), 0);
    auto sliceOffsets = mlir::SmallVector<int64_t>(inputShape.size(), 0);
    if (sliceOp != nullptr) {
        sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
    }
    const auto sliceChannelOffset = sliceOffsets[Dims4D::Act::C.ind()];

    if (sliceChannelOffset < inPadsEnd[Dim(expandDim)]) {
        padBegin[expandDim] = sliceChannelOffset;
        padEnd[expandDim] = inPadsEnd[Dim(expandDim)] - sliceChannelOffset;
    } else {
        padBegin[expandDim] = inPadsEnd[Dim(expandDim)];
    }

    return rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), expandValue,
                                               getIntArrayAttr(rewriter, ArrayRef(padBegin)),
                                               getIntArrayAttr(rewriter, ArrayRef(padEnd)));
}

mlir::Value paddingChannel(mlir::Operation* origOp, mlir::PatternRewriter& rewriter, mlir::Value expandValue,
                           ShapeRef padsEnd, size_t expandDim) {
    auto sliceOp = origOp->getOperand(0).getDefiningOp<IE::SliceOp>();
    if (sliceOp == nullptr) {
        return rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), expandValue, std::nullopt, padsEnd);
    }

    return expandWithOffset(rewriter, origOp, sliceOp, expandValue, padsEnd, expandDim);
}

mlir::Value paddingFilter(mlir::Operation* origOp, mlir::PatternRewriter& rewriter, mlir::Value expandValue,
                          Shape padsEnd) {
    auto sliceOp = origOp->getOperand(0).getDefiningOp<IE::SliceOp>();
    if (sliceOp == nullptr) {
        return rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), expandValue, std::nullopt, ShapeRef(padsEnd));
    }
    auto firstExpand = expandWithOffset(rewriter, origOp, sliceOp, expandValue, padsEnd, Dims4D::Act::C.ind());

    padsEnd[Dims4D::Filter::IC] = 0;
    return rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), firstExpand, std::nullopt, ShapeRef(padsEnd));
}

mlir::Value generateZeroConst(mlir::Location loc, vpux::NDTypeInterface inType, ShapeRef padShape,
                              mlir::PatternRewriter& rewriter) {
    const auto padType = inType.changeShape(ShapeRef(padShape));
    const auto eleType = padType.getElementType();

    const auto getEleStorageType = [&]() {
        if (const auto quantizedType = mlir::dyn_cast<mlir::quant::QuantizedType>(eleType)) {
            return normalizeQuantStorageType(quantizedType);
        } else {
            return eleType;
        }
    };
    const auto storageElementType = getEleStorageType();

    // FIXME: this is a very weird way to create 0 of a particular type
    auto outputBuffer = Const::Content::allocTempBuffer(padType, storageElementType, false);
    outputBuffer.fillWithZero();

    const auto dataType = mlir::cast<mlir::RankedTensorType>(padType.changeElemType(storageElementType));
    mlir::DenseElementsAttr eleAttr;
    const auto getDataAttr = [&](auto buffer) {
        eleAttr = Const::createConstContent(dataType, buffer);
    };
    outputBuffer.mutate(getDataAttr);

    return rewriter.create<Const::DeclareOp>(loc, padType, Const::ContentAttr::get(eleAttr)).getOutput();
}

mlir::Value concatWithZeroConst(mlir::Location loc, mlir::Value filter, ShapeRef subInput, int64_t sliceChannelOffset,
                                mlir::PatternRewriter& rewriter) {
    const auto filterType = mlir::cast<vpux::NDTypeInterface>(filter.getType());

    auto firstPadShape = to_small_vector(filterType.getShape());
    auto secondPadShape = to_small_vector(filterType.getShape());

    if (sliceChannelOffset <= subInput[Dims4D::Filter::IC]) {
        firstPadShape[Dims4D::Filter::IC.ind()] = sliceChannelOffset;
        secondPadShape[Dims4D::Filter::IC.ind()] = subInput[Dims4D::Filter::IC] - sliceChannelOffset;
    } else {
        firstPadShape[Dims4D::Filter::IC.ind()] = subInput[Dims4D::Filter::IC];
        secondPadShape[Dims4D::Filter::IC.ind()] = 0;
    }

    SmallVector<mlir::Value> concatInput;
    if (sliceChannelOffset > 0) {
        concatInput.push_back(generateZeroConst(loc, filterType, ShapeRef(firstPadShape), rewriter));
    }
    concatInput.push_back(filter);
    if (secondPadShape[Dims4D::Filter::IC.ind()] != 0) {
        concatInput.push_back(generateZeroConst(loc, filterType, ShapeRef(secondPadShape), rewriter));
    }
    auto concatOp = rewriter.create<IE::ConcatOp>(loc, concatInput, Dims4D::Filter::IC);

    return concatOp.getOutput();
}

mlir::Value padConvFilter(mlir::PatternRewriter& rewriter, mlir::Operation* origOp, const int64_t inChanPadEnd,
                          const int64_t outChanPadEnd, const Logger& log) {
    auto filterOperand = origOp->getOperand(1);
    if (inChanPadEnd == 0 && outChanPadEnd == 0) {
        return filterOperand;
    }

    auto inputOperand = origOp->getOperand(0);
    auto inputSliceOp = inputOperand.getDefiningOp<IE::SliceOp>();

    auto filterShape = mlir::cast<vpux::NDTypeInterface>(filterOperand.getType()).getShape();
    Shape filterPadsEnd(filterShape.size(), 0);
    filterPadsEnd[Dims4D::Filter::OC] = outChanPadEnd;
    filterPadsEnd[Dims4D::Filter::IC] = inChanPadEnd;

    auto filterOp = filterOperand.getDefiningOp();

    bool isConstFilter = mlir::isa_and_nonnull<Const::DeclareOp>(filterOp);
    if (!isConstFilter) {
        if (auto fqOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(filterOp)) {
            const auto fqInputConstOp = fqOp.getInput().getDefiningOp<Const::DeclareOp>();
            isConstFilter = fqInputConstOp != nullptr;
        }
    }

    // E#72287: Convert ExpandOp to const Concat in VPUIP, ExpandOp is preferred in IE for optimization.
    const auto expandTensor = [&](mlir::Value filter, ShapeRef pad, IE::SliceOp sliceOp) {
        auto sliceOffsets = mlir::SmallVector<int64_t>(pad.size(), 0);
        auto sliceChannelOffset = 0;
        if (sliceOp != nullptr) {
            sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
            sliceChannelOffset = sliceOffsets[Dims4D::Act::C.ind()];
        }
        return concatWithZeroConst(origOp->getLoc(), filter, pad, sliceChannelOffset, rewriter);
    };

    mlir::Value paddedFilter;
    if (!isConstFilter && inChanPadEnd != 0 && outChanPadEnd == 0) {
        // 1 dim expand for non-const filter
        log.trace("Pad non-const filter in IC at '{0}'", origOp->getLoc());
        paddedFilter = expandTensor(filterOperand, filterPadsEnd, inputSliceOp);

    } else if (!isConstFilter && inChanPadEnd != 0 && outChanPadEnd != 0) {
        // 2 dims expand for non-const filter
        log.trace("Pad non-const filter in IC & OC at '{0}'", origOp->getLoc());

        mlir::Value paddedFilter1;
        Shape filterPadsEnd1(filterShape.size(), 0);
        filterPadsEnd1[Dims4D::Filter::IC] = inChanPadEnd;
        paddedFilter1 = expandTensor(filterOperand, filterPadsEnd1, inputSliceOp);

        Shape filterPadsEnd2(filterShape.size(), 0);
        filterPadsEnd2[Dims4D::Filter::OC] = outChanPadEnd;
        paddedFilter = rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), paddedFilter1, std::nullopt,
                                                           ShapeRef(filterPadsEnd2));
    } else {
        // Const filter expand or expand on OC only
        paddedFilter = paddingFilter(origOp, rewriter, filterOperand, std::move(filterPadsEnd));
    }
    return paddedFilter;
}

// With respect to eltwise ops in a chain, for example:
//   Expand -> Add -> Slice -> Expand -> Add -> Slice
// It will be beneficial to keep the 2nd Expand for the 2nd Add instead of folding with Slice.
// So that the 2nd Add can utilize AdjustInputShapeForEltwise pass
bool beneficialToKeepExpand(ShapeRef unExpandedShape, ShapeRef expandedShape, mlir::Operation* childOp) {
    if (!childOp->hasOneUse()) {
        return false;
    }
    const auto isEltwiseOp = [](mlir::Operation* op) {
        if (op == nullptr) {
            return false;
        }
        // Mul/Sub/Add are selected since they are covered by the AdjustInputShapeForEltwise pass
        if (auto grpConvOp = mlir::dyn_cast<IE::GroupConvolutionOp>(op)) {
            return isEltwiseGroupConv(grpConvOp);
        } else if (mlir::isa<IE::MultiplyOp, IE::SubtractOp, IE::AddOp>(op)) {
            return true;
        }
        return false;
    };

    vpux::Logger log("beneficialToKeepExpand", vpux::LogLevel::Info);
    while (isEltwiseOp(childOp) && VPU::NCEInvariant::isSupported(childOp).succeeded()) {
        auto shapeCastResult = getShapeCastExpandedShape(childOp, expandedShape, unExpandedShape, log);
        if (mlir::failed(shapeCastResult)) {
            return false;
        }
        auto sliceChildOp = mlir::dyn_cast_or_null<IE::SliceOp>(*childOp->getResult(0).getUsers().begin());
        if (sliceChildOp == nullptr) {
            return true;
        }
        auto expandChildOp = mlir::dyn_cast_or_null<IE::ExpandOp>(*sliceChildOp->getResult(0).getUsers().begin());
        if (expandChildOp == nullptr) {
            return true;
        }
        childOp = *childOp->getResult(0).getUsers().begin();
        if (childOp == nullptr) {
            return true;
        } else if (!childOp->hasOneUse()) {
            return false;
        }
    }
    return false;
}

int64_t calculateAlignmentRequirementForExpandOpConversion(const vpux::NDTypeInterface expandInType) {
    const auto channelAlignment = VPU::NCEInvariant::getAlignment(expandInType.getElementType());
    const auto expandInChannels = expandInType.getShape()[Dims4D::Act::C];
    const auto leastChannelMultiple = std::lcm(channelAlignment, expandInChannels);
    return leastChannelMultiple / expandInChannels;
}

bool beneficialToPadHeight(IE::ExpandOp origOp) {
    const auto expandInType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto expandInShape = expandInType.getShape();
    const auto convolutionAlignment = IE::calculateAlignmentRequirementForExpandOpConversion(expandInType);

    // TODO (E#128867): need to repeat the experiment of E#118379 and E#120473 for the unaligned case (W%16 != 0), to
    // understand the correct tradeoff between expand to conv solution and dma solution (expected to be different due to
    // the overhead of padding) and correctly set up the constraints. Current constraints are only avoiding regressions
    // manifesting in CI, but they should be formally studied through measurements.
    if (expandInShape[Dims4D::Act::W] % convolutionAlignment != 0 &&
        ((expandInShape[Dims4D::Act::H] <= 48) || (expandInShape[Dims4D::Act::H] >= 1440) ||
         (expandInShape[Dims4D::Act::W] <= 128))) {
        return false;
    }

    return true;
}

bool beneficialToReshapeHeightToChannel(IE::ExpandOp origOp) {
    auto inShape = getShape(origOp.getInput());
    if (inShape[Dims4D::Act::W] != 1) {
        return false;
    }

    const auto expandInType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto convolutionAlignment = IE::calculateAlignmentRequirementForExpandOpConversion(expandInType);
    if (inShape[Dims4D::Act::H] % convolutionAlignment != 0) {
        return false;
    }

    constexpr int64_t THRESHOLD_FOR_BENEFICIAL_CONVERSION = 4096;
    auto outShape = getShape(origOp.getOutput());
    return outShape.totalSize() >= THRESHOLD_FOR_BENEFICIAL_CONVERSION;
}

bool beneficialToPadWidth(IE::ExpandOp origOp) {
    const auto expandInType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto expandInShape = expandInType.getShape();
    const auto expandInW = expandInShape[Dims4D::Act::W];
    const auto convolutionAlignment = IE::calculateAlignmentRequirementForExpandOpConversion(expandInType);

    //
    // Padding W to make Expand could convert to Conv
    // For example:
    //            Expand         :    1x1x32x79741 -> 1x16x32x79741
    //              |
    //            Conv
    // Convert to:
    //            Expand         :    1x1x32x79741 -> 1x1x32x79744
    //              |
    //            AffineReshape  :    1x1x32x79744 -> 1x16x32x4984xf16
    //              |
    //            New Conv
    //              |
    //            AffineReshape
    //              |
    //            Conv
    //
    if (expandInW % convolutionAlignment == 0) {
        return false;
    }

    if (!origOp.getResult().hasOneUse()) {
        return false;
    }

    mlir::Operation* userOp = *origOp.getResult().getUsers().begin();
    if (!mlir::isa<IE::ConvolutionOp, IE::TransposedConvolutionOp, IE::MaxPoolOp, IE::AvgPoolOp>(userOp)) {
        return false;
    }

    SmallVector<int64_t> padsBegin = {0, 0};
    SmallVector<int64_t> padsEnd = {0, 0};
    SmallVector<int64_t> stridesData = {1, 1};
    llvm::TypeSwitch<mlir::Operation*, void>(userOp)
            .Case<IE::ConvolutionOp, IE::TransposedConvolutionOp, IE::MaxPoolOp, IE::AvgPoolOp>([&](auto userOp) {
                padsBegin = parseIntArrayAttr<int64_t>(userOp.getPadsBeginAttr());
                padsEnd = parseIntArrayAttr<int64_t>(userOp.getPadsEndAttr());
                stridesData = parseIntArrayAttr<int64_t>(userOp.getStrides());
            });

    if (padsBegin.size() != 2 || padsBegin.size() != 2) {
        return false;
    }

    // Beneficial to pad W for case that the new width / strideX is same as original width then sliceOp is not required
    const auto alignedExpandInW = alignValUp(expandInShape[Dims4D::Act::W], convolutionAlignment);
    const auto inWWithPad =
            expandInW + padsBegin[Dims4D::PadsBegin::Left.ind()] + padsEnd[Dims4D::PadsEnd::Right.ind()];
    const auto alignedInWWithPad =
            alignedExpandInW + padsBegin[Dims4D::PadsBegin::Left.ind()] + padsEnd[Dims4D::PadsEnd::Right.ind()];
    return inWWithPad / stridesData[Dims4D::Strides::X.ind()] ==
           alignedInWWithPad / stridesData[Dims4D::Strides::X.ind()];
}

bool isEligibleConvertToConv(IE::ExpandOp expandOp, Logger log, StringRef debugName) {
    const auto expandInType = mlir::cast<vpux::NDTypeInterface>(expandOp.getInput().getType());
    const auto expandOutType = mlir::cast<vpux::NDTypeInterface>(expandOp.getOutput().getType());
    const auto supportedLayout = DimsOrder::NHWC;
    const auto expandInLayout = expandInType.getDimsOrder();
    if (expandInLayout != supportedLayout) {
        log.trace("[{0}]: Expand at {1} has {2} input layout, expected {3}", debugName, expandOp.getLoc(),
                  expandInLayout, supportedLayout);
        return false;
    }
    const auto expandOutLayout = expandOutType.getDimsOrder();
    if (expandOutLayout != supportedLayout) {
        log.trace("[{0}]: Expand at {1} has {2} output layout, expected {3}", debugName, expandOp.getLoc(),
                  expandOutLayout, supportedLayout);
        return false;
    }
    const auto expandPadsBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBeginAttr());
    if (expandPadsBegin.size() != 4) {
        log.trace("[{0}]: Expand at {1} has {2}-d start padding. Only 4-d shapes are supported", debugName,
                  expandOp.getLoc(), expandPadsBegin.size());
        return false;
    }
    const auto isConflictingPadBegin = [](const int64_t pad) -> bool {
        return pad != 0;
    };
    if (std::any_of(expandPadsBegin.begin(), expandPadsBegin.end(), isConflictingPadBegin)) {
        log.trace("[{0}]: Expand at {1} has {2} start padding. Expected to have [0, 0, 0, 0]", debugName,
                  expandOp.getLoc(), expandPadsBegin);
        return false;
    }
    const auto expandPadsEnd = parseIntArrayAttr<int64_t>(expandOp.getPadsEndAttr());
    if (expandPadsEnd.size() != 4) {
        log.trace("[{0}]: Expand at {1} has {2}-d end padding. Only 4-d shapes are supported", debugName,
                  expandOp.getLoc(), expandPadsEnd.size());
        return false;
    }
    if (expandPadsEnd[Dims4D::Act::N.ind()] != 0 || expandPadsEnd[Dims4D::Act::C.ind()] <= 0 ||
        expandPadsEnd[Dims4D::Act::H.ind()] != 0 || expandPadsEnd[Dims4D::Act::W.ind()] != 0) {
        log.trace("[{0}]: Expand at {1} has {2} end padding. Expected to have [0, C, 0, 0]", debugName,
                  expandOp.getLoc(), expandPadsEnd);
        return false;
    }
    const auto expandInShape = expandInType.getShape();
    if (expandInShape.size() != 4) {
        log.trace("[{0}]: Expand at {1} has {2}-d shape. Only 4-d shapes are supported", debugName, expandOp.getLoc(),
                  expandInShape.size());
        return false;
    }

    if (!expandInType.getElementType().isF16() &&
        !mlir::isa<mlir::quant::QuantizedType>(expandInType.getElementType())) {
        log.trace("[{0}]: Expand at {1} has {2} element type. Only float16 and quantized types are supported",
                  debugName, expandOp.getLoc(), expandInType.getElementType());
        return false;
    }

    // There are two conversion methods for Expand Op
    // 1. Convert to Convolution
    // 2. Convert to DMA
    // Experimental data shows the inference time is related to the channel-size and dimC/dimH(dimC/dimW).
    // Experimental Constraint E#118379:
    //    Utilize DMA conversion when channel size exceeds 32 (since expand with small channel will cause lots of stride
    //    DMAs, convert to conv is more efficient) and ratioH/ratioW exceeds 0.7 (since if channel is much smaller than
    //    height/wight also cause lots of stride DMAs).
    auto ratioH = static_cast<float>(expandInShape[Dims4D::Act::C]) / expandInShape[Dims4D::Act::H];
    auto ratioW = static_cast<float>(expandInShape[Dims4D::Act::C]) / expandInShape[Dims4D::Act::W];
    if (expandInShape[Dims4D::Act::C] > 32 && (ratioH > 0.7 || ratioW > 0.7)) {
        log.trace("[{0}]: Expansion at {1} has {2} channels exceeds '32', and spatial size (H {3} * W {4})."
                  "Converting to convolution is not beneficial",
                  debugName, expandOp.getLoc(), expandInShape[Dims4D::Act::C], expandInShape[Dims4D::Act::H],
                  expandInShape[Dims4D::Act::W]);
        return false;
    }

    const auto convolutionAlignment = IE::calculateAlignmentRequirementForExpandOpConversion(expandInType);
    const auto isToReshape = beneficialToReshapeHeightToChannel(expandOp);
    if (!isToReshape && !beneficialToPadHeight(expandOp) && !beneficialToPadWidth(expandOp)) {
        log.trace("[{0}]: Expand at {1} has shape {2} not compatible with alignment {3}. Expand to conv only for case "
                  "beneficialToReshapeHeightToChannel or beneficialToPadHeight or beneficialToPadWidth",
                  debugName, expandOp.getLoc(), expandInShape, convolutionAlignment);
        return false;
    }

    if (!isToReshape && expandInShape[Dims4D::Act::N] != 1) {
        log.trace("[{0}]: Expand at {1} has batch {2}. Expected to have 1", debugName, expandOp.getLoc(),
                  expandInShape[Dims4D::Act::N]);
        return false;
    }

    if (VPU::inputCompatibleWithAutoPad(expandInType) && VPU::hasAutoPaddingIDU(getModuleOp(expandOp))) {
        const auto hasConvUser = llvm::any_of(expandOp.getOutput().getUsers(), [](mlir::Operation* userOp) {
            return mlir::isa<IE::ConvolutionOp>(userOp);
        });
        if (hasConvUser) {
            log.trace("[{0}]: Expand at {1} has a candidate user which could use IDU autopad. Converting to "
                      "convolution is not beneficial",
                      debugName, expandOp.getLoc());
            return false;
        }
    }

    return true;
}

std::optional<vpux::Dim> getExpandAxis(IE::ExpandOp expandOp) {
    const auto expandAxes =
            vpux::IE::getDiffInOutSizeDims(getShape(expandOp.getInput()), getShape(expandOp.getResult()));
    if (expandAxes.empty() || expandAxes.size() != 1) {
        return std::nullopt;
    }
    return expandAxes.front();
}

}  // namespace IE
}  // namespace vpux
