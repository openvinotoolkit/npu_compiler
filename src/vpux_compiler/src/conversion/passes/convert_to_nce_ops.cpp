//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/attributes/arch.hpp"
#include "vpux/compiler/dialect/VPUIP/dpu_tiler.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// Utilities
//

const EnumMap<VPUIP::ArchKind, VPUIP::MPEMode> mpeMap = {
        {VPUIP::ArchKind::KMB, VPUIP::MPEMode::VECTOR_FP16},   //
        {VPUIP::ArchKind::TBH, VPUIP::MPEMode::VECTOR_FP16},   //
        {VPUIP::ArchKind::MTL, VPUIP::MPEMode::CUBOID_16x16},  //
};

mlir::Value createWeightsTableTensor(mlir::OpBuilder& builder, mlir::Location loc, int64_t OC, mlir::Value op_input,
                                     mlir::Value op_output, mlir::Value weights, mlir::Value bias,
                                     mlir::Value activationWindow) {
    SmallVector<int64_t> weightTableShape{OC, 1, 1, VPUIP::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC};

    auto* ctx = builder.getContext();

    const auto dataType = changeDimsOrder(mlir::MemRefType::get(weightTableShape, getSInt32Type(builder.getContext())),
                                          DimsOrder::NHWC);

    // const auto dataType = mlir::MemRefType::get(weightTableShape, getSInt32Type(builder.getContext()));
    auto createWeightsTableOp =
            builder.create<VPUIP::WeightsTableOp>(loc, dataType, op_input, op_output, weights, bias, activationWindow);

    const auto cmxMemSpaceAttr = VPUIP::PhysicalMemoryAttr::get(ctx, VPUIP::PhysicalMemory::CMX_NN);
    const auto dataTypeCMX = changeMemSpace(dataType, cmxMemSpaceAttr);

    auto dataAllocOp = builder.create<mlir::memref::AllocOp>(loc, dataTypeCMX);

    auto copyOp = builder.create<IERT::CopyOp>(loc, createWeightsTableOp.output(), dataAllocOp);

    return copyOp.output();
}

mlir::Value prepareTensorForDPU(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input) {
    // DMA DDR -> CMX
    const auto origType = input.getType().cast<mlir::MemRefType>();
    auto typeCMX = changeMemSpace(origType,
                                  VPUIP::PhysicalMemoryAttr::get(builder.getContext(), VPUIP::PhysicalMemory::CMX_NN));
    auto dmaAllocOp = builder.create<mlir::memref::AllocOp>(loc, typeCMX);
    auto dmaOp = builder.create<IERT::CopyOp>(loc, input, dmaAllocOp.memref());

    return dmaOp.output();
}

void addDPUTasks(VPUIP::NCEClusterTaskOp nceOp, mlir::PatternRewriter& rewriter, int64_t numDPU, int64_t opPadLeft,
                 int64_t opPadRight, int64_t opPadTop, int64_t opPadBottom, VPUIP::MPEMode mpeMode) {
    auto* ctx = nceOp.getContext();

    const auto outputShape = getShape(nceOp.output());
    numDPU = 1;  // TODO remove. Temp to enable comparison to MCM.
    const auto dpuTiles = VPUIP::DpuTiler::tileOverH(numDPU, outputShape, opPadLeft, opPadRight, opPadTop, opPadBottom);

    for (const auto& dpuTile : dpuTiles) {
        const auto startAttr = getIntArrayAttr(ctx, makeArrayRef(dpuTile.start));
        const auto endAttr = getIntArrayAttr(ctx, makeArrayRef(dpuTile.end));

        const auto pad =
                VPUIP::PaddingAttr::get(getIntAttr(ctx, dpuTile.padLeft), getIntAttr(ctx, dpuTile.padRight),
                                        getIntAttr(ctx, dpuTile.padTop), getIntAttr(ctx, dpuTile.padBottom), ctx);

        mpeMode = VPUIP::MPEMode::MATRIX;  // TODO remove. Temp to enable comparison to MCM.
        nceOp.addDPUTask(rewriter, startAttr, endAttr, pad, mpeMode);
    }
}

struct PostOpParams {
    VPUIP::PPELayerType layerType;
    int64_t clampLow;
    int64_t clampHigh;
    int64_t LreluMult;
    int64_t LreluShift;
};

static mlir::Optional<PostOpParams> parsePostOp(VPUIP::NCEClusterTaskOp nceOp, IE::PostOp postOp) {
    if (postOp == nullptr) {
        return mlir::None;
    }

    if (postOp.name().getValue() == IE::ReLUOp::getOperationName()) {
        VPUX_THROW_UNLESS(postOp.attrs().empty(), "'{0}' PostOp should not have any attributes", postOp.name());

        const int64_t clampLow = 0;
        const int64_t clampHigh = std::numeric_limits<int32_t>::max();
        const int64_t LreluMult = 1;
        const int64_t LreluShift = 0;
        return PostOpParams{VPUIP::PPELayerType::LRELU, clampLow, clampHigh, LreluMult, LreluShift};
    } else if (postOp.name().getValue() == IE::ClampOp::getOperationName()) {
        IE::ClampOp::Adaptor clamp(None, postOp.attrs());
        VPUX_THROW_UNLESS(clamp.verify(nceOp->getLoc()).succeeded(), "Wrong attributes '{0}' for '{1}' PostOp",
                          postOp.attrs(), postOp.name());

        const int64_t clampLow = vpux::toFixedPoint(clamp.min().getValueAsDouble());
        const int64_t clampHigh = vpux::toFixedPoint(clamp.max().getValueAsDouble());
        const int64_t LreluMult = 1;
        const int64_t LreluShift = 0;
        VPUX_THROW_UNLESS(clampLow == 0, "'{0}' PostOp with non-zero minimum is not supported", postOp.name());

        return PostOpParams{VPUIP::PPELayerType::LRELUX, clampLow, clampHigh, LreluMult, LreluShift};
    } else {
        VPUX_THROW("Unsupported PostOp '{0}'", postOp.name());
    }

    return mlir::None;
}

//
// ConvRewrite
//

//
// MaxPoolRewrite
//

class MaxPoolRewrite final : public mlir::OpRewritePattern<IERT::MaxPoolOp> {
public:
    MaxPoolRewrite(mlir::MLIRContext* ctx, int64_t numDPU, vpux::VPUIP::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IERT::MaxPoolOp>(ctx), _numDPU(numDPU), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const int64_t _numDPU;
    vpux::VPUIP::ArchKind _arch;
    Logger _log;
};

mlir::Value createActivationWindowTensor(mlir::OpBuilder& builder, mlir::Location loc, ArrayRef<uint8_t> fakeSparsity,
                                         int64_t numChannels) {
    auto* ctx = builder.getContext();
    const auto elemType = getUInt8Type(builder.getContext());

    //Needs to be different
    //SmallVector<int64_t> fakeSparsityShape{numChannels, 1, 1, static_cast<int64_t>(fakeSparsity.size()) / numChannels};
    SmallVector<int64_t> fakeSparsityShape{64, 1, 4, 16};

    const auto dataStorageType = mlir::RankedTensorType::get(fakeSparsityShape, elemType);
    const auto dataAttr = mlir::DenseElementsAttr::get(dataStorageType, fakeSparsity);

    const auto dataType = mlir::MemRefType::get(fakeSparsityShape, elemType);
    auto dataConstOp = builder.create<Const::DeclareOp>(loc, dataType, Const::ContentAttr::get(dataAttr));

    const auto cmxMemSpaceAttr = VPUIP::PhysicalMemoryAttr::get(ctx, VPUIP::PhysicalMemory::CMX_NN);
    const auto dataTypeCMX = changeMemSpace(dataType, cmxMemSpaceAttr);

    auto dataAllocOp = builder.create<mlir::memref::AllocOp>(loc, dataTypeCMX);
    auto copyOp = builder.create<IERT::CopyOp>(loc, dataConstOp.output(), dataAllocOp);

    return copyOp.output();
}

class ConvRewrite final : public mlir::OpRewritePattern<IERT::ConvolutionOp> {
public:
    ConvRewrite(mlir::MLIRContext* ctx, int64_t numDPU, vpux::VPUIP::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IERT::ConvolutionOp>(ctx), _numDPU(numDPU), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const int64_t _numDPU;
    vpux::VPUIP::ArchKind _arch;
    Logger _log;
};

static mlir::Value alignchannelMajorWeightTensor(mlir::OpBuilder& builder, mlir::Location loc,
                                                 const mlir::Value origFilter) {
    const auto filterShape = getShape(origFilter);
    const auto OC = filterShape[IE::Dims4D::Filter::OC];
    const auto filtersPerInChan = filterShape[IE::Dims4D::Filter::IC];
    const auto KY = filterShape[IE::Dims4D::Filter::KY];
    const auto KX = filterShape[IE::Dims4D::Filter::KX];

    const auto origFilterType = origFilter.getType().cast<mlir::ShapedType>();
    const auto channelMajorConvAlignment = VPUIP::NCEInvariant::getChannelAlignment(origFilterType.getElementType());
    const int64_t remainder = (filtersPerInChan * KY * KX) % channelMajorConvAlignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);
    if (remainder == 0) {
        // nothing to align
        return origFilter;
    }

    auto weightsConst = origFilter.getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(weightsConst != nullptr, "Grouped convolution does not provide constant weights");

    const int64_t alignment = channelMajorConvAlignment - remainder;
    auto weightsContentAttr = weightsConst.contentAttr();
    auto nchwWeightsContentAttr = weightsContentAttr.reorder(DimsOrder::NCHW);

    auto flatWeightShape = Shape{OC, 1, 1, filtersPerInChan * KY * KX};
    auto flatWeightsContentAttr = nchwWeightsContentAttr.reshape(flatWeightShape);
    auto alignedWeightsContentAttr = flatWeightsContentAttr.padWithZero({0, 0, 0, 0}, {0, 0, 0, alignment});
    auto nhwcWeightsContentAttr = alignedWeightsContentAttr.reorder(DimsOrder::NHWC);

    auto alignedWeightShape = SmallVector<int64_t>{OC, 1, 1, filtersPerInChan * KY * KX + alignment};
    const auto outAllocType = mlir::MemRefType::get(alignedWeightShape, origFilterType.getElementType());
    const auto outAllocTypeNHWC = changeDimsOrder(outAllocType, DimsOrder::NHWC);

    auto alignedWeightsOp = builder.create<Const::DeclareOp>(loc, outAllocTypeNHWC, nhwcWeightsContentAttr);

    return alignedWeightsOp.output();
}

mlir::LogicalResult ConvRewrite::matchAndRewrite(IERT::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const {
    if (VPUIP::NCEInvariant::verifyOp(origOp, _log).failed()) {
        return matchFailed(rewriter, origOp, "Operation {0} does not satisfy the NCE invariant", origOp);
    }

    //
    // Get dimensions
    //

    const auto filterShape = getShape(origOp.filter());

    const auto IC = filterShape[IE::Dims4D::Filter::IC];
    const auto OC = filterShape[IE::Dims4D::Filter::OC];
    const auto KY = filterShape[IE::Dims4D::Filter::KY];
    const auto KX = filterShape[IE::Dims4D::Filter::KX];

    //
    // Prepare input for DPU
    //

    auto inputDPU = prepareTensorForDPU(rewriter, origOp->getLoc(), origOp.input());
    auto alignedFilter = alignchannelMajorWeightTensor(rewriter, origOp->getLoc(), origOp.filter());
    auto filterDPU = prepareTensorForDPU(rewriter, origOp->getLoc(), alignedFilter);

    //
    // Generate activation window
    //

    const auto origInputType = origOp.input().getType().cast<mlir::MemRefType>();
    // FIXME why does fake sparsity expects this order of kernel dimensions?
    const auto kernelSize = SmallVector<int64_t>{KX, KY};
    const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto bitPatternSize =
            VPUIP::NCESparsity::getBitPatternSize(kernelSize, kernelStrides[0], origInputType.getElementType());

    const auto actWindowChanLen = getIntAttr(getContext(), bitPatternSize);
    const auto fakeSparsity =
            VPUIP::NCESparsity::getFakeSparsity(kernelSize, kernelStrides[0], origInputType.getElementType(), IC, OC);
    const auto activationWindow = createActivationWindowTensor(rewriter, origOp->getLoc(), fakeSparsity, OC);

    //
    // Prepare output buffer for DPU
    //

    const auto origOutType = origOp.output().getType().cast<mlir::MemRefType>();
    const auto outReorderType = changeDimsOrder(origOutType, DimsOrder::NHWC);
    const auto outTypeCMX =
            changeMemSpace(outReorderType, VPUIP::PhysicalMemoryAttr::get(getContext(), VPUIP::PhysicalMemory::CMX_NN));

    auto outAllocOpCMX = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), outTypeCMX);

    auto weightsTable = createWeightsTableTensor(rewriter, origOp->getLoc(), OC, inputDPU, outAllocOpCMX.memref(),
                                                 filterDPU, origOp.bias(), activationWindow);

    //
    // Create NCE per-cluster Operation
    //

    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    const auto kernelPaddingAttr =
            getIntArrayAttr(getContext(), makeArrayRef({padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0]}));

    const auto kernelSizeAttr = getIntArrayAttr(getContext(), makeArrayRef({KY, KX}));

    Logger::global().error("order: {0}", DimsOrder::fromValue(origOp.input()));

    if (DimsOrder::NCHW == DimsOrder::fromValue(origOp.input())) {
        auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(
                origOp->getLoc(), inputDPU, filterDPU, weightsTable, /*activation_window=*/nullptr,
                /*parent_input=*/inputDPU,
                /*parent_output=*/outAllocOpCMX.memref(),
                /*output_buff=*/outAllocOpCMX.memref(), VPUIP::NCETaskType::CMCONV, kernelSizeAttr, origOp.strides(),
                kernelPaddingAttr, actWindowChanLen, /*is_continued*/ nullptr);

        addDPUTasks(nceOp, rewriter, _numDPU, padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0], mpeMap.at(_arch));

        const auto postOpParams = parsePostOp(nceOp, origOp.post_opAttr());
        if (postOpParams.hasValue()) {
            nceOp.addPPETask(rewriter, postOpParams->layerType, postOpParams->clampLow, postOpParams->clampHigh,
                             postOpParams->LreluMult, postOpParams->LreluShift);
        }

        //
        // DMA output CMX -> DDR
        //

        rewriter.replaceOpWithNewOp<IERT::CopyOp>(origOp, nceOp.output(), origOp.output_buff());
    } else if (DimsOrder::NHWC == DimsOrder::fromValue(origOp.input())) {
        auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(
                origOp->getLoc(), inputDPU, filterDPU, weightsTable, /*activation_window=*/nullptr,
                /*parent_input=*/inputDPU,
                /*parent_output=*/outAllocOpCMX.memref(),
                /*output_buff=*/outAllocOpCMX.memref(), VPUIP::NCETaskType::CONV, kernelSizeAttr, origOp.strides(),
                kernelPaddingAttr, /*activation_window_channel_length=*/nullptr, /*is_continued*/ nullptr);

        addDPUTasks(nceOp, rewriter, _numDPU, padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0], mpeMap.at(_arch));
        const auto postOpParams = parsePostOp(nceOp, origOp.post_opAttr());
        if (postOpParams.hasValue()) {
            nceOp.addPPETask(rewriter, postOpParams->layerType, postOpParams->clampLow, postOpParams->clampHigh,
                             postOpParams->LreluMult, postOpParams->LreluShift);
        }

        //
        // DMA output CMX -> DDR
        //

        rewriter.replaceOpWithNewOp<IERT::CopyOp>(origOp, nceOp.output(), origOp.output_buff());
    }

    return mlir::success();
}

mlir::LogicalResult MaxPoolRewrite::matchAndRewrite(IERT::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    if (VPUIP::NCEInvariant::verifyOp(origOp, _log).failed()) {
        return matchFailed(rewriter, origOp, "Operation {0} does not satisfy the NCE invariant", origOp);
    }

    //
    // Get dimensions
    //

    const auto origInputType = origOp.input().getType().cast<mlir::MemRefType>();
    const auto inputShape = getShape(origInputType);

    const auto IC = inputShape[IE::Dims4D::Act::C];

    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.strides());

    const auto bitPatternSize =
            VPUIP::NCESparsity::getBitPatternSize(kernelSize, kernelStrides[0], origInputType.getElementType());

    //
    // Prepare input for DPU
    //

    auto inputDPU = prepareTensorForDPU(rewriter, origOp->getLoc(), origOp.input());

    //
    // Generate activation window
    //

    const auto fakeSparsity =
            VPUIP::NCESparsity::getFakeSparsity(kernelSize, kernelStrides[0], origInputType.getElementType(), IC, IC);
    const auto activationWindow = createActivationWindowTensor(rewriter, origOp->getLoc(), fakeSparsity, IC);

    //
    // Prepare output buffer for DPU
    //

    const auto origOutType = origOp.output().getType().cast<mlir::MemRefType>();
    const auto outReorderType = changeDimsOrder(origOutType, DimsOrder::NHWC);
    const auto outTypeCMX =
            changeMemSpace(outReorderType, VPUIP::PhysicalMemoryAttr::get(getContext(), VPUIP::PhysicalMemory::CMX_NN));

    auto outAllocOpCMX = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), outTypeCMX);

    auto weightsTable = createWeightsTableTensor(rewriter, origOp->getLoc(), IC, inputDPU, outAllocOpCMX.memref(),
                                                 nullptr, nullptr, activationWindow);

    //
    // Create NCE per-cluster Operation
    //

    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    const auto kernelPaddingAttr =
            getIntArrayAttr(getContext(), makeArrayRef({padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0]}));

    const auto activation_window_channel_length = getIntAttr(getContext(), static_cast<uint32_t>(bitPatternSize));

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(
            origOp->getLoc(), inputDPU, /*weights=*/nullptr, weightsTable, activationWindow,
            /*parent_input=*/inputDPU,
            /*parent_output=*/outAllocOpCMX.memref(),
            /*output_buff=*/outAllocOpCMX.memref(), VPUIP::NCETaskType::MAXPOOL, origOp.kernel_size(), origOp.strides(),
            kernelPaddingAttr, activation_window_channel_length, /*is_continued*/ nullptr);

    addDPUTasks(nceOp, rewriter, _numDPU, padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0], mpeMap.at(_arch));
    const auto postOpParams = parsePostOp(nceOp, origOp.post_opAttr());
    if (postOpParams.hasValue()) {
        nceOp.addPPETask(rewriter, postOpParams->layerType, postOpParams->clampLow, postOpParams->clampHigh,
                         postOpParams->LreluMult, postOpParams->LreluShift);
    }

    //
    // DMA output CMX -> DDR
    //

    rewriter.replaceOpWithNewOp<IERT::CopyOp>(origOp, nceOp.output(), origOp.output_buff());

    return mlir::success();
}

//
// EltwiseAddRewrite
//

class EltwiseAddRewrite final : public mlir::OpRewritePattern<IERT::AddOp> {
public:
    EltwiseAddRewrite(mlir::MLIRContext* ctx, int64_t numDPU, vpux::VPUIP::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IERT::AddOp>(ctx), _numDPU(numDPU), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const int64_t _numDPU;
    vpux::VPUIP::ArchKind _arch;
    Logger _log;
};

mlir::LogicalResult EltwiseAddRewrite::matchAndRewrite(IERT::AddOp origOp, mlir::PatternRewriter& rewriter) const {
    if (VPUIP::NCEInvariant::verifyOp(origOp, _log).failed()) {
        return matchFailed(rewriter, origOp, "Operation {0} does not satisfy the NCE invariant", origOp);
    }

    //
    // Prepare input for DPU
    //

    auto firstInputDPU = prepareTensorForDPU(rewriter, origOp->getLoc(), origOp.input1());
    auto secondInputDPU = prepareTensorForDPU(rewriter, origOp->getLoc(), origOp.input2());

    //
    // Prepare output buffer for DPU
    //

    const auto origOutType = origOp.output().getType().cast<mlir::MemRefType>();
    const auto outReorderType = changeDimsOrder(origOutType, DimsOrder::NHWC);
    const auto outTypeCMX =
            changeMemSpace(outReorderType, VPUIP::PhysicalMemoryAttr::get(getContext(), VPUIP::PhysicalMemory::CMX_NN));

    auto outAllocOpCMX = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), outTypeCMX);

    //
    // Create NCE per-cluster Operation
    //

    const auto activation_window_channel_length = getIntAttr(getContext(), static_cast<int32_t>(0));

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(origOp->getLoc(), firstInputDPU, secondInputDPU,
                                                          /*weightsTable=*/nullptr,
                                                          /*activation_window=*/nullptr,
                                                          /*parent_input=*/firstInputDPU,
                                                          /*parent_output=*/outAllocOpCMX.memref(),
                                                          /*output_buff=*/outAllocOpCMX.memref(),
                                                          VPUIP::NCETaskType::ELTWISE,
                                                          /*kernel_size=*/nullptr,
                                                          /*kernel_strides=*/nullptr,
                                                          /*kernel_padding=*/nullptr, activation_window_channel_length,
                                                          /*is_continued*/ nullptr);

    int64_t clampLow = std::numeric_limits<int32_t>::min();
    int64_t clampHigh = std::numeric_limits<int32_t>::max();
    int64_t LreluMult = 1;
    int64_t LreluShift = 0;
    const auto postOpParams = parsePostOp(nceOp, origOp.post_opAttr());
    if (postOpParams.hasValue()) {
        clampLow = postOpParams->clampLow;
        clampHigh = postOpParams->clampHigh;
        LreluMult = postOpParams->LreluMult;
        LreluShift = postOpParams->LreluShift;
    }
    nceOp.addPPETask(rewriter, VPUIP::PPELayerType::ADD, clampLow, clampHigh, LreluMult, LreluShift);

    //
    // Create DPU sub-task
    //

    const SmallVector<int64_t> padsBegin = {0, 0};
    const SmallVector<int64_t> padsEnd = {0, 0};
    addDPUTasks(nceOp, rewriter, _numDPU, padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0], mpeMap.at(_arch));

    //
    // DMA output CMX -> DDR
    //

    rewriter.replaceOpWithNewOp<IERT::CopyOp>(origOp, nceOp.output(), origOp.output_buff());

    return mlir::success();
}

//
// DepthwiseConvRewrite
//

class DepthwiseConvRewrite final : public mlir::OpRewritePattern<IERT::GroupConvolutionOp> {
public:
    DepthwiseConvRewrite(mlir::MLIRContext* ctx, int64_t numDPU, vpux::VPUIP::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<IERT::GroupConvolutionOp>(ctx), _numDPU(numDPU), _arch(arch), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IERT::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const int64_t _numDPU;
    vpux::VPUIP::ArchKind _arch;
    Logger _log;
};

static mlir::Value alignDepthwiseWeightTensor(mlir::OpBuilder& builder, mlir::Location loc,
                                              const mlir::Value origFilter) {
    const auto filterShape = getShape(origFilter);
    const auto OC = filterShape[IE::Dims4D::Filter::OC];
    const auto filtersPerInChan = filterShape[IE::Dims4D::Filter::IC];
    const auto KY = filterShape[IE::Dims4D::Filter::KY];
    const auto KX = filterShape[IE::Dims4D::Filter::KX];

    const auto origFilterType = origFilter.getType().cast<mlir::ShapedType>();
    const auto depthwiseConvAlignment = VPUIP::NCEInvariant::getChannelAlignment(origFilterType.getElementType());
    const int64_t remainder = (filtersPerInChan * KY * KX) % depthwiseConvAlignment;
    VPUX_THROW_UNLESS(remainder >= 0, "Channel alignment cannot be negative: {0}", remainder);
    if (remainder == 0) {
        // nothing to align
        return origFilter;
    }

    auto weightsConst = origFilter.getDefiningOp<Const::DeclareOp>();
    VPUX_THROW_UNLESS(weightsConst != nullptr, "Grouped convolution does not provide constant weights");

    const int64_t alignment = depthwiseConvAlignment - remainder;
    auto weightsContentAttr = weightsConst.contentAttr();
    auto nchwWeightsContentAttr = weightsContentAttr.reorder(DimsOrder::NCHW);

    auto flatWeightShape = Shape{OC, filtersPerInChan * KY * KX, 1, 1};
    auto flatWeightsContentAttr = nchwWeightsContentAttr.reshape(flatWeightShape);
    auto alignedWeightsContentAttr = flatWeightsContentAttr.padWithZero({0, 0, 0, 0}, {0, alignment, 0, 0});
    auto nhwcWeightsContentAttr = alignedWeightsContentAttr.reorder(DimsOrder::NHWC);

    auto alignedWeightShape = SmallVector<int64_t>{OC, filtersPerInChan * KY * KX + alignment, 1, 1};
    const auto outAllocType = mlir::MemRefType::get(alignedWeightShape, origFilterType.getElementType());
    const auto outAllocTypeNHWC = changeDimsOrder(outAllocType, DimsOrder::NHWC);
    auto alignedWeightsOp = builder.create<Const::DeclareOp>(loc, outAllocTypeNHWC, nhwcWeightsContentAttr);

    return alignedWeightsOp.output();
}

mlir::LogicalResult DepthwiseConvRewrite::matchAndRewrite(IERT::GroupConvolutionOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    if (VPUIP::NCEInvariant::verifyOp(origOp, _log).failed()) {
        return matchFailed(rewriter, origOp, "Operation {0} does not satisfy the NCE invariant", origOp);
    }

    //
    // Get dimensions
    //

    const auto filterShape = getShape(origOp.filter());

    const auto IC = filterShape[IE::Dims4D::Filter::IC];
    const auto OC = filterShape[IE::Dims4D::Filter::OC];
    const auto KY = filterShape[IE::Dims4D::Filter::KY];
    const auto KX = filterShape[IE::Dims4D::Filter::KX];

    //
    // Prepare input for DPU
    //

    auto inputDPU = prepareTensorForDPU(rewriter, origOp->getLoc(), origOp.input());

    auto alignedFilter = alignDepthwiseWeightTensor(rewriter, origOp->getLoc(), origOp.filter());
    auto filterDPU = prepareTensorForDPU(rewriter, origOp->getLoc(), alignedFilter);

    //
    // Generate activation window
    //

    const auto origInputType = origOp.input().getType().cast<mlir::MemRefType>();
    // FIXME why does fake sparsity expects this order of kernel dimensions?
    const auto kernelSize = SmallVector<int64_t>{KX, KY};
    const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.strides());
    const auto bitPatternSize =
            VPUIP::NCESparsity::getBitPatternSize(kernelSize, kernelStrides[0], origInputType.getElementType());
    const auto actWindowChanLen = getIntAttr(getContext(), bitPatternSize);

    const auto fakeSparsity =
            VPUIP::NCESparsity::getFakeSparsity(kernelSize, kernelStrides[0], origInputType.getElementType(), IC, OC);
    const auto activationWindow = createActivationWindowTensor(rewriter, origOp->getLoc(), fakeSparsity, OC);

    //
    // Prepare output buffer for DPU
    //

    const auto origOutType = origOp.output().getType().cast<mlir::MemRefType>();
    const auto outReorderType = changeDimsOrder(origOutType, DimsOrder::NHWC);
    const auto outTypeCMX =
            changeMemSpace(outReorderType, VPUIP::PhysicalMemoryAttr::get(getContext(), VPUIP::PhysicalMemory::CMX_NN));

    auto outAllocOpCMX = rewriter.create<mlir::memref::AllocOp>(origOp->getLoc(), outTypeCMX);

    auto weightsTable = createWeightsTableTensor(rewriter, origOp->getLoc(), OC, inputDPU, outAllocOpCMX.memref(),
                                                 filterDPU, origOp.bias(), activationWindow);

    //
    // Create NCE per-cluster Operation
    //

    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    const auto kernelPaddingAttr =
            getIntArrayAttr(getContext(), makeArrayRef({padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0]}));

    const auto kernelSizeAttr = getIntArrayAttr(getContext(), makeArrayRef({KY, KX}));

    auto nceOp = rewriter.create<VPUIP::NCEClusterTaskOp>(
            origOp->getLoc(), inputDPU, filterDPU, weightsTable, activationWindow,
            /*parent_input=*/inputDPU,
            /*parent_output=*/outAllocOpCMX.memref(),
            /*output_buff=*/outAllocOpCMX.memref(), VPUIP::NCETaskType::DWCONV, kernelSizeAttr, origOp.strides(),
            kernelPaddingAttr, actWindowChanLen, /*is_continued*/ nullptr);

    addDPUTasks(nceOp, rewriter, _numDPU, padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0], mpeMap.at(_arch));
    const auto postOpParams = parsePostOp(nceOp, origOp.post_opAttr());
    if (postOpParams.hasValue()) {
        nceOp.addPPETask(rewriter, postOpParams->layerType, postOpParams->clampLow, postOpParams->clampHigh,
                         postOpParams->LreluMult, postOpParams->LreluShift);
    }

    //
    // DMA output CMX -> DDR
    //

    rewriter.replaceOpWithNewOp<IERT::CopyOp>(origOp, nceOp.output(), origOp.output_buff());

    return mlir::success();
}

//
// ConvertToNCEOpsPass
//

class ConvertToNCEOpsPass final : public ConvertToNCEOpsBase<ConvertToNCEOpsPass> {
public:
    ConvertToNCEOpsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConvertToNCEOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();

    auto module = func->getParentOfType<mlir::ModuleOp>();

    const auto arch = VPUIP::getArch(module);
    VPUX_THROW_UNLESS(mpeMap.find(arch) != mpeMap.end(), "Failed to map MPE mode to target arch");

    auto resOp = IERT::RunTimeResourcesOp::getFromModule(module);
    VPUX_THROW_UNLESS(resOp != nullptr, "Missing IERT run-time resources definition");

    auto nceCluster = resOp.getExecutor(VPUIP::PhysicalProcessorAttr::get(&ctx, VPUIP::PhysicalProcessor::NCE_Cluster));
    VPUX_THROW_UNLESS(nceCluster != nullptr, "Failed to get NCE_Cluster information");

    auto dpuExec = nceCluster.getSubExecutor(
            VPUIP::PhysicalProcessorAttr::get(&ctx, VPUIP::PhysicalProcessor::NCE_PerClusterDPU));
    VPUX_THROW_UNLESS(dpuExec != nullptr, "Failed to get DPU information");

    mlir::OwningRewritePatternList patterns(&ctx);
    patterns.insert<ConvRewrite>(&ctx, dpuExec.count(), arch, _log);
    patterns.insert<MaxPoolRewrite>(&ctx, dpuExec.count(), arch, _log);
    patterns.insert<EltwiseAddRewrite>(&ctx, dpuExec.count(), arch, _log);
    patterns.insert<DepthwiseConvRewrite>(&ctx, dpuExec.count(), arch, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::createConvertToNCEOpsPass(Logger log) {
    return std::make_unique<ConvertToNCEOpsPass>(log);
}
