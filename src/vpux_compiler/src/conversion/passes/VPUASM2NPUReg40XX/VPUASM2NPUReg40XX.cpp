//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <sys/types.h>
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/composers/dma_composer.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/utils.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"

#include "vpux/utils/core/error.hpp"

#include <npu_40xx_nnrt.hpp>

namespace vpux {
#define GEN_PASS_DECL_CONVERTVPUASM2NPUREG40XX
#define GEN_PASS_DEF_CONVERTVPUASM2NPUREG40XX
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;
using namespace vpux::VPURegMapped;
using namespace npu40xx;
using namespace NPUReg40XX;
using namespace NPUReg40XX::Descriptors;

namespace {

class NNDMARewriter final : public mlir::OpRewritePattern<VPUASM::NNDMAOp> {
public:
    NNDMARewriter(mlir::MLIRContext* ctx, Logger log, ELF::SymbolReferenceMap& symRefMap)
            : mlir::OpRewritePattern<VPUASM::NNDMAOp>(ctx), _log(log), _symRefMap(symRefMap) {
        setDebugName("NNDMA_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::NNDMAOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    ELF::SymbolReferenceMap& _symRefMap;
};

mlir::LogicalResult NNDMARewriter::matchAndRewrite(VPUASM::NNDMAOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto dma = rewriter.create<NPUReg40XX::NNDMAOp>(
            origOp->getLoc(), origOp.getSymNameAttr(),
            DMARegisterAttr::get(rewriter.getContext(), DMADescriptorComposer::compose(origOp, _symRefMap)),
            origOp.getInputAttr(), origOp.getOutputBuffsAttr(), origOp.getNextLinkAttr(),
            origOp.getActCompressionSizeEntryAttr(), origOp.getIndicesAttr());

    // TODO: (E#114625) Remove once proper refactoring happened
    if (!origOp.getTaskLocationAttr()) {
        dma.getOperation()->setAttr("directLink", rewriter.getUnitAttr());
    }

    rewriter.eraseOp(origOp);

    return mlir::success();
}

void setNormFactor(VpuMediaTask& initValues, ::mlir::ArrayAttr normFactor) {
    const auto getRawFP16 = [](auto val) {
        const auto valFP16 = vpux::type::float16(val);
        return valFP16.to_bits();
    };

    auto normArr = parseFPArrayAttr<double>(normFactor);
    VPUX_THROW_UNLESS(normArr.size() == MEDIA_MAX_NUM_PLANES * 4 /*MEDIA_MAX_NUM_NORM_FACT*/,
                      "Normalization array is invalid");

    initValues.write<Registers::NormFactor_0, Fields::NormFact0>(getRawFP16(normArr[0]));
    initValues.write<Registers::NormFactor_0, Fields::NormFact1>(getRawFP16(normArr[1]));
    initValues.write<Registers::NormFactor_0, Fields::NormFact2>(getRawFP16(normArr[2]));
    initValues.write<Registers::NormFactor_0, Fields::NormFact3>(getRawFP16(normArr[3]));

    initValues.write<Registers::NormFactor_1, Fields::NormFact0>(getRawFP16(normArr[4]));
    initValues.write<Registers::NormFactor_1, Fields::NormFact1>(getRawFP16(normArr[5]));
    initValues.write<Registers::NormFactor_1, Fields::NormFact2>(getRawFP16(normArr[6]));
    initValues.write<Registers::NormFactor_1, Fields::NormFact3>(getRawFP16(normArr[7]));

    initValues.write<Registers::NormFactor_2, Fields::NormFact0>(getRawFP16(normArr[8]));
    initValues.write<Registers::NormFactor_2, Fields::NormFact1>(getRawFP16(normArr[9]));
    initValues.write<Registers::NormFactor_2, Fields::NormFact2>(getRawFP16(normArr[10]));
    initValues.write<Registers::NormFactor_2, Fields::NormFact3>(getRawFP16(normArr[11]));
}

uint8_t getBytesOfPackOfPixels(VPU::M2iColorFmt inFormat) {
    switch (inFormat) {
    case VPU::M2iColorFmt::PL_FP16_RGB:
    case VPU::M2iColorFmt::PL_FP16_YUV:
    case VPU::M2iColorFmt::SP_NV12_10:
    case VPU::M2iColorFmt::SP_P010:
        return 2;
    case VPU::M2iColorFmt::IL_RGB888:
        return 3;
    case VPU::M2iColorFmt::IL_RGB8888:
    case VPU::M2iColorFmt::IL_RGB30:
        return 4;
    default:
        return 1;
    };
}

void setMediaDimensions(VPUASM::DeclareBufferOp bufferOp, VPU::M2iColorFmt format, uint64_t& width, uint64_t& height) {
    auto elemShape = bufferOp.getBufferType().getMemref().cast<NDTypeInterface>().getShape();

    switch (format) {
    case VPU::M2iColorFmt::PL_YUV420_8:
    case VPU::M2iColorFmt::SP_NV12_8:  // dims[] = N(0),H(1),W(2),C(3)
        // H / 3 * 2 -- These YUV formats have a full sized Y plane, and weaved U,V values,
        // hence we need to extract the height of the Y plane from the concatenated height
        height = elemShape[Dims4D::Act::C] / 3 * 2;
        width = elemShape[Dims4D::Act::H];
        break;

    case VPU::M2iColorFmt::IL_RGB888:  // dims[] = N(0),H(1),W(2),C(3)
        height = elemShape[Dims4D::Act::C];
        width = elemShape[Dims4D::Act::H];
        break;

    case VPU::M2iColorFmt::PL_RGB24:     // dims[] = N(0),C(1),H(2),W(3)
    case VPU::M2iColorFmt::PL_FP16_RGB:  // dims[] = N(0),C(1),H(2),W(3)
        height = elemShape[Dims4D::Act::H];
        width = elemShape[Dims4D::Act::W];
        break;

    default:
        VPUX_THROW("{0} format is not supported", format);
        break;
    }
}

void setInSizeDescription(VpuMediaTask& initValues, VPU::M2iColorFmt inFormat, uint64_t width, uint64_t height,
                          uint64_t m2iIndex) {
    uint64_t inSize0_ls(0), PSOB_inPS(0), inSize1_width(0), inSize1_height(0);
    uint64_t inSize1_ls(0), inSize2_width(0), inSize2_height(0), inSize2_ls(0);

    auto inSize0_width = width - 1;
    auto inSize0_height = height - 1;

    auto inSize0_PID = m2iIndex;

    switch (inFormat) {
    case VPU::M2iColorFmt::PL_RGB24:
    case VPU::M2iColorFmt::PL_YUV444_8:
        inSize0_ls = width;
        PSOB_inPS = width * height;
        inSize1_width = width - 1;
        inSize1_height = height - 1;
        inSize1_ls = width;
        inSize2_width = width - 1;
        inSize2_height = height - 1;
        inSize2_ls = width;
        break;

    case VPU::M2iColorFmt::PL_FP16_RGB:
        inSize0_ls = width * 2;
        PSOB_inPS = width * height * 2;
        inSize1_width = width - 1;
        inSize1_height = height - 1;
        inSize1_ls = width * 2;
        inSize2_width = width - 1;
        inSize2_height = height - 1;
        inSize2_ls = width * 2;
        break;

    case VPU::M2iColorFmt::PL_GRAY8:
        inSize0_ls = width;
        PSOB_inPS = width * height;
        inSize1_width = width - 1;
        inSize1_height = height - 1;
        inSize1_ls = width;
        inSize2_width = width - 1;
        inSize2_height = height - 1;
        inSize2_ls = width;
        break;

    case VPU::M2iColorFmt::SP_NV12_8:
        inSize0_ls = width;
        PSOB_inPS = width * height;
        inSize1_width = width - 1;
        inSize1_height = height / 2 - 1;
        inSize1_ls = width;
        break;

    case VPU::M2iColorFmt::PL_YUV420_8:
        inSize0_ls = width;
        PSOB_inPS = width * height;
        inSize1_width = width / 2 - 1;
        inSize1_height = height / 2 - 1;
        inSize1_ls = width / 2;
        inSize2_width = width / 2 - 1;
        inSize2_height = height / 2 - 1;
        inSize2_ls = width / 2;
        break;

    case VPU::M2iColorFmt::PL_YUV422_8:
        inSize0_ls = width;
        PSOB_inPS = width * height;
        inSize1_width = width / 2 - 1;
        inSize1_height = height - 1;
        inSize1_ls = width / 2;
        inSize2_width = width / 2 - 1;
        inSize2_height = height - 1;
        inSize2_ls = width / 2;
        break;

    case VPU::M2iColorFmt::IL_RGB888:
        inSize0_ls = width * 3;
        PSOB_inPS = width * height * 3;
        inSize1_width = width - 1;
        inSize1_height = height - 1;
        inSize1_ls = width * 3;
        inSize2_width = width - 1;
        inSize2_height = height - 1;
        inSize2_ls = width * 3;
        break;

    default:
        VPUX_THROW("invalid input format {0}", inFormat);
        break;
    }

    initValues.write<Registers::inSize0, Fields::ls>(inSize0_ls);
    initValues.write<Registers::inSize0, Fields::width>(inSize0_width);
    initValues.write<Registers::inSize0, Fields::height>(inSize0_height);
    initValues.write<Fields::pid>(inSize0_PID);

    initValues.write<Registers::inSize1, Fields::ls>(inSize1_ls);
    initValues.write<Registers::inSize1, Fields::width>(inSize1_width);
    initValues.write<Registers::inSize1, Fields::height>(inSize1_height);

    initValues.write<Registers::inSize2, Fields::ls>(inSize2_ls);
    initValues.write<Registers::inSize2, Fields::width>(inSize2_width);
    initValues.write<Registers::inSize2, Fields::height>(inSize2_height);

    initValues.write<Fields::inPS>(PSOB_inPS);
}

void setOutDescription(VpuMediaTask& initValues, VPU::M2iColorFmt outFormat, uint64_t outWidth, uint64_t outHeight) {
    uint64_t outScale0_width(0), outScale0_height(0);
    uint64_t psSc0Y(0), psSc0UV(0), lsSc0Y(0), lsSc0UV(0);

    switch (outFormat) {
    case VPU::M2iColorFmt::PL_RGB24:
    case VPU::M2iColorFmt::PL_GRAY8:
        outScale0_width = outWidth - 1;
        outScale0_height = outHeight - 1;
        psSc0Y = outWidth * outHeight;
        lsSc0Y = outWidth;
        break;

    case VPU::M2iColorFmt::PL_FP16_YUV:
    case VPU::M2iColorFmt::PL_FP16_RGB:
        outScale0_width = outWidth - 1;
        outScale0_height = outHeight - 1;
        psSc0Y = outWidth * outHeight * 2;
        lsSc0Y = outWidth * 2;
        break;

    case VPU::M2iColorFmt::SP_NV12_8:
        outScale0_width = outWidth - 1;
        outScale0_height = outHeight - 1;
        psSc0Y = outWidth * outHeight;
        psSc0UV = outWidth * outHeight / 2;
        lsSc0Y = outWidth;
        lsSc0UV = outWidth;
        break;

    case VPU::M2iColorFmt::PL_YUV420_8:
        outScale0_width = outWidth - 1;
        outScale0_height = outHeight - 1;
        psSc0Y = outWidth * outHeight;
        psSc0UV = outWidth * outHeight / 4;
        lsSc0Y = outWidth;
        lsSc0UV = outWidth / 2;
        break;

    case VPU::M2iColorFmt::PL_YUV422_8:
        outScale0_width = outWidth - 1;
        outScale0_height = outHeight - 1;
        psSc0Y = outWidth * outHeight;
        psSc0UV = outWidth * outHeight / 2;
        lsSc0Y = outWidth;
        lsSc0UV = outWidth / 2;
        break;

    case VPU::M2iColorFmt::PL_YUV444_8:
        outScale0_width = outWidth - 1;
        outScale0_height = outHeight - 1;
        psSc0Y = outWidth * outHeight;
        psSc0UV = outWidth * outHeight;
        lsSc0Y = outWidth;
        lsSc0UV = outWidth;
        break;

    case VPU::M2iColorFmt::IL_RGB888:
        outScale0_width = outWidth - 1;
        outScale0_height = outHeight - 1;
        psSc0Y = outWidth * outHeight * 3;
        lsSc0Y = outWidth * 3;
        break;

    default:
        VPUX_THROW("invalid output format {0}", outFormat);
        break;
    }

    initValues.write<Fields::outScale0_width>(outScale0_width);
    initValues.write<Fields::outScale0_height>(outScale0_height);
    initValues.write<Fields::psSc0Y>(psSc0Y);
    initValues.write<Fields::psSc0UV>(psSc0UV);
    initValues.write<Fields::lsSc0Y>(lsSc0Y);
    initValues.write<Fields::lsSc0UV>(lsSc0UV);
}

bool isCscRequired(VPU::M2iColorFmt inFormat, VPU::M2iColorFmt outFormat) {
    // Automatically switch CSC on when input format and output format are different
    // and they are found in a viable conversion list
    llvm::DenseMap<VPU::M2iColorFmt, llvm::DenseSet<VPU::M2iColorFmt>> supportedInOutFormatMap = {
            {VPU::M2iColorFmt::SP_NV12_8,
             {VPU::M2iColorFmt::PL_RGB24, VPU::M2iColorFmt::IL_RGB888, VPU::M2iColorFmt::PL_FP16_RGB}},
            {VPU::M2iColorFmt::PL_RGB24,
             {VPU::M2iColorFmt::SP_NV12_8, VPU::M2iColorFmt::PL_YUV444_8, VPU::M2iColorFmt::PL_YUV422_8,
              VPU::M2iColorFmt::PL_GRAY8, VPU::M2iColorFmt::PL_YUV420_8}},
            {VPU::M2iColorFmt::IL_RGB888, {VPU::M2iColorFmt::SP_NV12_8}},
            {VPU::M2iColorFmt::PL_YUV444_8, {VPU::M2iColorFmt::PL_RGB24}},
            {VPU::M2iColorFmt::PL_YUV422_8, {VPU::M2iColorFmt::PL_RGB24}},
            {VPU::M2iColorFmt::PL_YUV420_8,
             {VPU::M2iColorFmt::PL_RGB24, VPU::M2iColorFmt::IL_RGB888, VPU::M2iColorFmt::PL_FP16_RGB}}};

    return (supportedInOutFormatMap.find(inFormat) != supportedInOutFormatMap.end() &&
            supportedInOutFormatMap[inFormat].find(outFormat) != supportedInOutFormatMap[inFormat].end());
}

class M2IRewriter final : public mlir::OpRewritePattern<VPUASM::M2IOp> {
public:
    M2IRewriter(mlir::MLIRContext* ctx, Logger log, ELF::SymbolReferenceMap& symRefMap)
            : mlir::OpRewritePattern<VPUASM::M2IOp>(ctx), _log(log), _symRefMap(symRefMap) {
        setDebugName("M2I_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::M2IOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    ELF::SymbolReferenceMap& _symRefMap;
};

mlir::LogicalResult M2IRewriter::matchAndRewrite(VPUASM::M2IOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // // prepare MediaRegister
    VpuMediaTask descriptor;

    auto outFormat = static_cast<uint64_t>(origOp.getOutFmt());
    auto sampleType = static_cast<uint64_t>(origOp.getInterp());

    const auto chromaInRC = static_cast<uint64_t>(origOp.getChromaInReverseChannels());
    const auto lumaInRC = static_cast<uint64_t>(origOp.getLumaInReverseChannels());
    auto ifc = ((chromaInRC & 0x1) << 5) | ((lumaInRC & 0x1) << 4) | (getBytesOfPackOfPixels(origOp.getInFmt()) & 0xF);
    uint64_t irqMask = 1 << 15;

    const auto chromaOutRC = static_cast<uint64_t>(origOp.getChromaOutReverseChannels());
    const auto lumaOutRC = static_cast<uint64_t>(origOp.getLumaOutReverseChannels());
    auto ofc = ((chromaOutRC & 0x1) << 1) | (lumaOutRC & 0x1);

    uint64_t nextDescTileMask = 0;
    if (origOp.getNextLink().has_value()) {
        auto nextM2IRef = _symRefMap.lookupSymbol(origOp.getNextLink().value());
        if (auto nextM2ITaskBuffer = mlir::dyn_cast<VPUASM::DeclareTaskBufferOp>(nextM2IRef)) {
            nextDescTileMask = NPUReg40XX::getTileSelectMaskForBuffer(nextM2ITaskBuffer);
        }
    }

    uint64_t width(0), height(0), inputTileMask(0);
    auto m2iIndex = origOp.getTaskIndex().getValue();
    auto inBufferRef = _symRefMap.lookupSymbol(origOp.getInput());
    auto inBufferOp = mlir::dyn_cast_or_null<VPUASM::DeclareBufferOp>(inBufferRef);
    VPUX_THROW_UNLESS(inBufferOp, "Could not find symbol name entry for {0}", inBufferRef);
    inputTileMask = NPUReg40XX::getTileSelectMaskForBuffer(inBufferOp);
    setMediaDimensions(inBufferOp, origOp.getInFmt(), width, height);
    setInSizeDescription(descriptor, origOp.getInFmt(), width, height, m2iIndex);
    auto roiWidth = width - 1;
    auto roiHeight = height - 1;

    uint64_t outWidth(0), outHeight(0), outputTileMask(0);
    auto outBufferRef = _symRefMap.lookupSymbol(origOp.getOutputBuff());
    auto outBufferOp = mlir::dyn_cast_or_null<VPUASM::DeclareBufferOp>(outBufferRef);
    VPUX_THROW_UNLESS(outBufferOp, "Could not find symbol name entry for {0}", outBufferRef);
    outputTileMask = NPUReg40XX::getTileSelectMaskForBuffer(outBufferOp);
    setMediaDimensions(outBufferOp, origOp.getOutFmt(), outWidth, outHeight);
    setOutDescription(descriptor, origOp.getOutFmt(), outWidth, outHeight);
    outWidth = outWidth - 1;
    outHeight = outHeight - 1;

    if (origOp.getNorm().has_value()) {
        setNormFactor(descriptor, origOp.getNorm().value());
    }

    uint64_t operations(0);
    operations |= origOp.getDoCsc() ? (1 << 0) : 0;
    operations |= isCscRequired(origOp.getInFmt(), origOp.getOutFmt()) ? (1 << 3 | 1 << 0) : 0;  // CLAMP bit always set
    operations |= origOp.getDoNorm() ? 1 << 1 : 0;

    descriptor.write<Registers::inAddr0, Fields::inAddr>(inputTileMask);
    descriptor.write<Registers::inAddr1, Fields::inAddr>(inputTileMask);
    descriptor.write<Registers::inAddr2, Fields::inAddr>(inputTileMask);
    descriptor.write<Fields::inFormat>(static_cast<uint64_t>(origOp.getInFmt()));
    descriptor.write<Fields::outFormat>(outFormat);
    descriptor.write<Fields::sampleType>(sampleType);
    descriptor.write<Fields::numRois>(1);
    descriptor.write<Fields::IFC>(ifc);
    descriptor.write<Fields::IRQMask>(irqMask);
    descriptor.write<Fields::operations>(operations);
    descriptor.write<Fields::roiBase>(outputTileMask);
    descriptor.write<Fields::OFC>(ofc);
    descriptor.write<Fields::outFormatLocal>(outFormat);
    descriptor.write<Fields::samlingTypeLocal>(sampleType);
    descriptor.write<Fields::outScale0_width>(outWidth);
    descriptor.write<Fields::outScale0_height>(outHeight);
    descriptor.write<Fields::roiWidth>(roiWidth);
    descriptor.write<Fields::roiHeight>(roiHeight);
    descriptor.write<Fields::vSc_offset>(origOp.getTileOffsetY().value_or(0));
    descriptor.write<Fields::hSc_offset>(origOp.getTileOffsetX().value_or(0));
    descriptor.write<Fields::vSc_factor>(origOp.getScaleFactorY());
    descriptor.write<Fields::hSc_factor>(origOp.getScaleFactorX());
    descriptor.write<Fields::nextDesc>(nextDescTileMask);
    descriptor.write<Fields::barGateMaskLO>(VPUMI40XX::computeMaskLo(origOp.getWaitBarriers()));
    descriptor.write<Fields::barGateMaskHI>(VPUMI40XX::computeMaskHi(origOp.getWaitBarriers()));
    descriptor.write<Fields::barUpdateLO>(VPUMI40XX::computeMaskLo(origOp.getUpdateBarriers()));
    descriptor.write<Fields::barUpdateHI>(VPUMI40XX::computeMaskHi(origOp.getUpdateBarriers()));
    descriptor.write<Registers::media_barriers_sched_, Fields::start_after_>(origOp.getStartAfter());
    descriptor.write<Registers::media_barriers_sched_, Fields::clean_after_>(origOp.getCleanAfter());

    auto regM2IDescriptorAttr = VpuMediaTaskAttr::get(rewriter.getContext(), std::move(descriptor));

    rewriter.create<NPUReg40XX::M2IOp>(origOp->getLoc(), origOp.getSymNameAttr(), origOp.getInputAttr(),
                                       origOp.getOutputBuffAttr(), origOp.getProfilingDataAttr(),
                                       origOp.getNextLinkAttr(), regM2IDescriptorAttr);

    rewriter.eraseOp(origOp);

    return mlir::success();
}

//
// ActShave
//

class ActShaveRtRewriter final : public mlir::OpRewritePattern<VPUASM::ActShaveRtOp> {
public:
    ActShaveRtRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUASM::ActShaveRtOp>(ctx), _log(log) {
        setDebugName("ActShaveRt_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::ActShaveRtOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ActShaveRtRewriter::matchAndRewrite(VPUASM::ActShaveRtOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    rewriter.create<NPUReg40XX::ActShaveRtOp>(origOp->getLoc(), origOp.getSymNameAttr(), origOp.getKernelPathAttr());
    rewriter.eraseOp(origOp);
    return mlir::success();
}

//
// ActKernelRange
//
class ActKernelRangeRewriter final : public mlir::OpRewritePattern<VPUASM::ActKernelRangeOp> {
public:
    ActKernelRangeRewriter(mlir::MLIRContext* ctx, Logger log, ELF::SymbolReferenceMap& symRefMap)
            : mlir::OpRewritePattern<VPUASM::ActKernelRangeOp>(ctx), _log(log), _symRefMap(symRefMap) {
        setDebugName("ActKernelRange_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::ActKernelRangeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    ELF::SymbolReferenceMap& _symRefMap;
};

mlir::LogicalResult ActKernelRangeRewriter::matchAndRewrite(VPUASM::ActKernelRangeOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto kernelEntry = NPUReg40XX::getKernelEntry(_symRefMap, origOp.getKernelEntry());
    auto kernelTextSize = NPUReg40XX::getKernelTextSize(_symRefMap, origOp.getKernelText());
    auto kernelTaskType = origOp.getKernelTaskType();
    auto kernelPath = NPUReg40XX::getKernelPath(_symRefMap, origOp.getKernelEntry(), kernelTaskType);
    auto actWLtype = static_cast<std::underlying_type<npu40xx::nn_public::VpuActWLType>::type>(
            NPUReg40XX::getActWLType(kernelTaskType));

    VpuActKernelRange descriptor;
    descriptor.write<Fields::type>(actWLtype);
    descriptor.write<Fields::kernel_entry>(kernelEntry);
    descriptor.write<Fields::code_size>(kernelTextSize);

    auto regActKernelDescriptorAttr = VpuActKernelRangeAttr::get(rewriter.getContext(), std::move(descriptor));

    rewriter.create<NPUReg40XX::ActKernelRangeOp>(origOp->getLoc(), origOp.getSymNameAttr(), regActKernelDescriptorAttr,
                                                  origOp.getTaskLocationAttr(), origOp.getKernelTextAttr(),
                                                  origOp.getKernelEntryAttr());

    _log.trace("[{0}] Got kernel '{1}' and cpu '{2}'", getDebugName(), kernelPath, VPU::getArch(origOp));

    rewriter.eraseOp(origOp);

    return mlir::success();
}

//
// ActKernelInvocation
//
class ActKernelInvocationRewriter final : public mlir::OpRewritePattern<VPUASM::ActKernelInvocationOp> {
public:
    ActKernelInvocationRewriter(mlir::MLIRContext* ctx, Logger log, ELF::SymbolReferenceMap& symRefMap)
            : mlir::OpRewritePattern<VPUASM::ActKernelInvocationOp>(ctx), _log(log), _symRefMap(symRefMap) {
        setDebugName("ActKernelInvocation_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::ActKernelInvocationOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    ELF::SymbolReferenceMap& _symRefMap;
};

mlir::LogicalResult ActKernelInvocationRewriter::matchAndRewrite(VPUASM::ActKernelInvocationOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto kernelRangeRef = _symRefMap.lookupSymbol(origOp.getKernelRange());
    auto kernelRangeTaskBufferOp = mlir::cast<VPUASM::DeclareTaskBufferOp>(kernelRangeRef);
    auto kernelRangeTileMask = NPUReg40XX::getTileSelectMaskForBuffer(kernelRangeTaskBufferOp);
    auto kernelRangeIndex = origOp.getRangeIndex();

    uint64_t perfPacketTileMask = 0;
    if (auto profilingDataOpt = origOp.getProfilingData()) {
        auto perfPacketBufferRef = _symRefMap.lookupSymbol(*profilingDataOpt);
        auto perfPacketBufferOp = mlir::cast<VPUASM::DeclareBufferOp>(perfPacketBufferRef);
        perfPacketTileMask = NPUReg40XX::getTileSelectMaskForBuffer(perfPacketBufferOp);
    }

    auto waitMaskHi = VPUMI40XX::computeMaskHi(origOp.getWaitBarriers());
    auto waitMaskLo = VPUMI40XX::computeMaskLo(origOp.getWaitBarriers());
    auto postMaskHi = VPUMI40XX::computeMaskHi(origOp.getUpdateBarriers());
    auto postMaskLo = VPUMI40XX::computeMaskLo(origOp.getUpdateBarriers());

    uint8_t barrier_group = 0;
    uint8_t barrier_mask = 0;

    std::tie(barrier_group, barrier_mask) = ELF::reduceWaitMaskTo8bit(waitMaskLo);

    auto nextAkiTileMask = origOp.getNextLink().has_value() ? kernelRangeTileMask : 0;

    VpuActKernelInvocation descriptor;
    descriptor.write<Fields::range>(kernelRangeTileMask);
    descriptor.write<Fields::barriers_wait_mask_hi_act>(waitMaskHi);
    descriptor.write<Fields::barriers_wait_mask_lo_act>(waitMaskLo);
    descriptor.write<Fields::barriers_post_mask_hi_act>(postMaskHi);
    descriptor.write<Fields::barriers_post_mask_lo_act>(postMaskLo);
    descriptor.write<Fields::group_act>(barrier_group);
    descriptor.write<Fields::mask_act>(barrier_mask);
    descriptor.write<Registers::act_invo_barriers_sched, Fields::start_after_>(origOp.getStartAfter());
    descriptor.write<Registers::act_invo_barriers_sched, Fields::clean_after_>(origOp.getCleanAfter());
    descriptor.write<Fields::invo_tile>(origOp.getTile());
    descriptor.write<Fields::kernel_range_index>(kernelRangeIndex);
    descriptor.write<Fields::perf_packet_out>(perfPacketTileMask);
    descriptor.write<Fields::next_aki_wl_addr>(nextAkiTileMask);

    auto regActKernelInvoDescriptorAttr = VpuActKernelInvocationAttr::get(rewriter.getContext(), std::move(descriptor));

    rewriter.create<NPUReg40XX::ActKernelInvocationOp>(
            origOp->getLoc(), origOp.getSymNameAttr(), regActKernelInvoDescriptorAttr, origOp.getTaskLocationAttr(),
            origOp.getNextLinkAttr(), origOp.getKernelRangeAttr(), origOp.getKernelDataAttr(),
            origOp.getKernelParamsAttr(), origOp.getProfilingDataAttr());

    rewriter.eraseOp(origOp);

    return mlir::success();
}

//
// NNRTConfig
//
class NNRTConfigRewriter final : public mlir::OpRewritePattern<VPUASM::NNrtConfigOp> {
public:
    NNRTConfigRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUASM::NNrtConfigOp>(ctx), _log(log) {
        setDebugName("NNRTConfig_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::NNrtConfigOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult NNRTConfigRewriter::matchAndRewrite(VPUASM::NNrtConfigOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    rewriter.create<NPUReg40XX::NNrtConfigOp>(
            origOp.getLoc(), origOp.getSymNameAttr(), origOp.getIsActKernelInvocations(), origOp.getActShaveRtAttr(),
            origOp.getActShaveStacksAttr(), origOp.getDmaHwpBaseAttr(), origOp.getHwpWorkpointCfgAttr());
    rewriter.eraseOp(origOp);
    return mlir::success();
}

//
// ManagedBarrier
//

class ManagedBarrierRewriter final : public mlir::OpRewritePattern<VPUASM::ManagedBarrierOp> {
public:
    ManagedBarrierRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUASM::ManagedBarrierOp>(ctx), _log(log) {
        setDebugName("ManagedBarrier_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::ManagedBarrierOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ManagedBarrierRewriter::matchAndRewrite(VPUASM::ManagedBarrierOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto workItemIdx = origOp.getWorkItemIdx();

    uint32_t workItemRegVal = 0;
    uint32_t enqueueCount = 0;

    if (workItemIdx.has_value()) {
        enqueueCount = origOp.getWorkItemCount();
        workItemRegVal = workItemIdx.value().getValue();
    }

    VpuTaskBarrierMap taskBarrierMapDescriptor;
    taskBarrierMapDescriptor.write<Fields::tb_next_same_id>(
            checked_cast_reg<NPUReg40XX::RegField_next_same_id_Type>(static_cast<uint32_t>(origOp.getNextSameId())));
    taskBarrierMapDescriptor.write<Fields::tb_producer_count>(origOp.getProducerCount());
    taskBarrierMapDescriptor.write<Fields::tb_consumer_count>(origOp.getConsumerCount());
    taskBarrierMapDescriptor.write<Fields::tb_real_id>(origOp.getId());
    taskBarrierMapDescriptor.write<Fields::tb_work_item_idx>(workItemRegVal);
    taskBarrierMapDescriptor.write<Fields::tb_enqueue_count>(enqueueCount);

    auto taskBarrierMapDescriptorAttr =
            VpuTaskBarrierMapAttr::get(rewriter.getContext(), std::move(taskBarrierMapDescriptor));

    rewriter.create<NPUReg40XX::ManagedBarrierOp>(origOp.getLoc(), origOp.getSymNameAttr(),
                                                  taskBarrierMapDescriptorAttr);
    rewriter.eraseOp(origOp);

    return mlir::success();
}

//
// BarrierConfigure
//
class BarrierRewriter final : public mlir::OpRewritePattern<VPUASM::ConfigureBarrierOp> {
public:
    BarrierRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUASM::ConfigureBarrierOp>(ctx), _log(log) {
        setDebugName("ConfigureBarrier_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::ConfigureBarrierOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult BarrierRewriter::matchAndRewrite(VPUASM::ConfigureBarrierOp origOp,
                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    // origOp.getNextSameId() is int64 with invalid barrier represented by -1 and a max
    // value of numeric_limits<uint32_t>::max() - 1
    // At this point it is cast to uint32 as required by the NNRuntime with invalid barrier
    // represented by numeric_limits<uint32_t>::max()
    VpuBarrierCountConfig barrierConfigDescriptor;
    barrierConfigDescriptor.write<Fields::next_same_id_>(
            checked_cast_reg<NPUReg40XX::RegField_next_same_id_Type>(static_cast<uint32_t>(origOp.getNextSameId())));
    barrierConfigDescriptor.write<Fields::producer_count_>(origOp.getProducerCount());
    barrierConfigDescriptor.write<Fields::consumer_count_>(origOp.getConsumerCount());
    barrierConfigDescriptor.write<Fields::real_id_>(origOp.getId());

    auto barrierConfigDescriptorAttr =
            VpuBarrierCountConfigAttr::get(rewriter.getContext(), std::move(barrierConfigDescriptor));
    rewriter.create<NPUReg40XX::ConfigureBarrierOp>(origOp->getLoc(), origOp.getSymNameAttr(),
                                                    barrierConfigDescriptorAttr);

    rewriter.eraseOp(origOp);

    return mlir::success();
}

//
// MappedInference
//

class MappedInferenceRewriter final : public mlir::OpRewritePattern<VPUASM::MappedInferenceOp> {
public:
    MappedInferenceRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUASM::MappedInferenceOp>(ctx), _log(log) {
        setDebugName("MappedInference_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::MappedInferenceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MappedInferenceRewriter::matchAndRewrite(VPUASM::MappedInferenceOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto dmaCount = parseIntArrayOfArrayAttr<int64_t>(origOp.getDmaCount());

    mlir::SmallVector<int64_t> dmaCountDDR;
    mlir::SmallVector<int64_t> dmaCountCMX;
    dmaCountDDR.reserve(dmaCount.size());
    dmaCountCMX.reserve(dmaCount.size());

    for (size_t dmaTileIndex = 0; dmaTileIndex < dmaCount.size(); dmaTileIndex++) {
        VPUX_THROW_UNLESS(dmaCount[dmaTileIndex].size() == 2, "Unsupported number of DMA types - '{0}'",
                          dmaCount[dmaTileIndex].size());

        dmaCountDDR.push_back(dmaCount[dmaTileIndex][static_cast<size_t>(VPUMI40XX::DmaNnSrcType::DDR)]);
        dmaCountCMX.push_back(dmaCount[dmaTileIndex][static_cast<size_t>(VPUMI40XX::DmaNnSrcType::CMX_NN)]);
    }

    const auto dmaCountDDRAttr = getIntArrayAttr(origOp.getContext(), ArrayRef(dmaCountDDR));
    const auto dmaCountCMXAttr = getIntArrayAttr(origOp.getContext(), ArrayRef(dmaCountCMX));

    rewriter.create<NPUReg40XX::MappedInferenceOp>(origOp->getLoc(),                           //
                                                   origOp.getSymNameAttr(),                    //
                                                   origOp.getDmaCountAttr(),                   //
                                                   dmaCountDDRAttr,                            //
                                                   dmaCountCMXAttr,                            //
                                                   origOp.getInvariantCountAttr(),             //
                                                   origOp.getVariantCountAttr(),               //
                                                   origOp.getActKernelRangesCountAttr(),       //
                                                   origOp.getActKernelInvocationsCountAttr(),  //
                                                   origOp.getMediaCountAttr(),                 //
                                                   origOp.getBarrierCountAttr(),               //
                                                   origOp.getMappedInferenceVersionAttr(),     //
                                                   origOp.getActShaveRtAttr(),                 //
                                                   origOp.getActShaveStacksAttr(),             //
                                                   origOp.getDmaHwpBaseAttr(),                 //
                                                   origOp.getHwpWorkpointCfgAttr(),            //
                                                   origOp.getManagedMappedInferenceAttr(),
                                                   origOp.getDmaTasksAttr(),              //
                                                   origOp.getInvariantTasksAttr(),        //
                                                   origOp.getVariantTasksAttr(),          //
                                                   origOp.getActKernelRangesAttr(),       //
                                                   origOp.getActKernelInvocationsAttr(),  //
                                                   origOp.getMediaTasksAttr(),            //
                                                   origOp.getBarrierTasksAttr());         //
    rewriter.eraseOp(origOp);

    return mlir::success();
}

//
// ManagedMappedInference
//

class ManagedMappedInferenceRewriter final : public mlir::OpRewritePattern<VPUASM::ManagedMappedInferenceOp> {
public:
    ManagedMappedInferenceRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUASM::ManagedMappedInferenceOp>(ctx), _log(log) {
        setDebugName("ManagedMappedInference_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::ManagedMappedInferenceOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    enum class DmaNnSrcType { DDR, CMX_NN, Count };
    Logger _log;
};

mlir::LogicalResult ManagedMappedInferenceRewriter::matchAndRewrite(VPUASM::ManagedMappedInferenceOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    VpuManagedMappedInference managedMappedInferenceDescriptor;
    managedMappedInferenceDescriptor.write<Fields::MMI_final_barrier>(origOp.getFinalBarrierId());
    managedMappedInferenceDescriptor.write<Fields::taskReferenceCount_MMI_work_item>(origOp.getWorkItemsCount());
    managedMappedInferenceDescriptor.write<Fields::taskReferenceCount_MMI_task_configs>(origOp.getBarrierCount());
    managedMappedInferenceDescriptor.write<Fields::taskReferenceCount_MMI_initial_barriers>(
            origOp.getBootstrapTasksCount());
    managedMappedInferenceDescriptor.write<Fields::MMI_bootstrap_workitems_count>(origOp.getBootsrapWorkItemsCount());
    managedMappedInferenceDescriptor.write<Fields::MMI_actshv_used>(origOp.getActshvUsed());
    managedMappedInferenceDescriptor.write<Fields::MMI_dpu_used>(origOp.getDpuUsed());
    managedMappedInferenceDescriptor.write<Fields::MMI_media_used>(origOp.getMediaUsed());
    managedMappedInferenceDescriptor.write<Fields::MMI_dma_from_ddr_used>(origOp.getDmaFromDdrUsed());
    managedMappedInferenceDescriptor.write<Fields::MMI_dma_from_cmx_used>(origOp.getDmaFromCmxUsed());
    managedMappedInferenceDescriptor.write<Fields::taskReferenceCount_MMI_nnrt_config>(1);
    managedMappedInferenceDescriptor.write<Fields::taskReferenceCount_MMI_barriers_configuration>(
            origOp.getBarrierConfigurationTasksCount());
    managedMappedInferenceDescriptor.write<Fields::taskReferenceCount_MMI_num_of_barrier_reprogrammings>(
            origOp.getBarriersReprogrammingCount());

    if (managedMappedInferenceDescriptor.read<Fields::taskReferenceCount_MMI_barriers_configuration>() > 0) {
        managedMappedInferenceDescriptor.write<Fields::MMI_barrier_programming_mode>(
                npu40xx::nn_public::VpuManagedMappedInference::VpuBarrierProgrammingMode::NO_BARRIER_DMAS_SCHEDULED);
    }

    managedMappedInferenceDescriptor.write<Fields::MMI_barrier_configuration_stride>(
            origOp.getBarrierConfigurationStride());

    auto managedMappedInferenceDescriptorAttr =
            VpuManagedMappedInferenceAttr::get(rewriter.getContext(), std::move(managedMappedInferenceDescriptor));

    rewriter.create<NPUReg40XX::ManagedMappedInferenceOp>(origOp->getLoc(),                            //
                                                          origOp.getSymNameAttr(),                     //
                                                          origOp.getNnrtConfigAttr(),                  //
                                                          origOp.getMappedInferenceVersionAttr(),      //
                                                          origOp.getDmaTasksAttr(),                    //
                                                          origOp.getWorkItemsAttr(),                   //
                                                          origOp.getBarrierTasksAttr(),                //
                                                          origOp.getBootstrapTasksAttr(),              //
                                                          origOp.getBarrierConfigurationTasksAttr(),   //
                                                          origOp.getNumOfBarrierReprogrammingsAttr(),  //
                                                          managedMappedInferenceDescriptorAttr);       //
    rewriter.eraseOp(origOp);

    return mlir::success();
}

//
// MappedInferenceVersion
//

class MappedInferenceVersionRewriter final : public mlir::OpRewritePattern<VPUASM::MappedInferenceVersionOp> {
public:
    MappedInferenceVersionRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUASM::MappedInferenceVersionOp>(ctx), _log(log) {
        setDebugName("MappedInferenceVersion_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::MappedInferenceVersionOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MappedInferenceVersionRewriter::matchAndRewrite(VPUASM::MappedInferenceVersionOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    rewriter.replaceOpWithNewOp<NPUReg40XX::MappedInferenceVersionOp>(
            origOp, origOp.getSymNameAttr(), origOp.getMajorAttr(), origOp.getMinorAttr(), origOp.getPatchAttr());
    return mlir::success();
}

//
// WorkItem
//

class WorkItemRewriter final : public mlir::OpRewritePattern<VPUASM::WorkItemOp> {
public:
    WorkItemRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUASM::WorkItemOp>(ctx), _log(log) {
        setDebugName("WorkItem_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::WorkItemOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult WorkItemRewriter::matchAndRewrite(VPUASM::WorkItemOp origOp,
                                                      mlir::PatternRewriter& rewriter) const {
    enum TaskType : uint8_t { DPU = 0, DMA, KERNEL, SYSTEM_MANAGEMENT, UNKNOWN = 255 };

    auto realTaskIndex = origOp.getRealTaskIndex();
    auto nextWorkItemIdx = origOp.getNextWorkitemIdx();

    uint32_t nextWorkItemIdxValue = 0;

    if (nextWorkItemIdx.has_value()) {
        nextWorkItemIdxValue = nextWorkItemIdx.value().getValue();
    }

    uint64_t descPtrOffset = 0;
    TaskType workItemType;

    switch (origOp.getTaskType()) {
    case VPURegMapped::TaskType::DPUVariant:
        workItemType = TaskType::DPU;
        descPtrOffset = static_cast<uint64_t>(VPUMI40XX::generateTileMask({realTaskIndex.getTileIdx()}));
        break;
    case VPURegMapped::TaskType::DMA:
        workItemType = TaskType::DMA;
        break;
    case VPURegMapped::TaskType::ActKernelInvocation:
        workItemType = TaskType::KERNEL;
        descPtrOffset = static_cast<uint64_t>(VPUMI40XX::generateTileMask({realTaskIndex.getTileIdx()}));
        break;
    default:
        return origOp.emitOpError("Invalid workItem task type");
    }

    WorkItem workItemDescriptor;
    workItemDescriptor.write<Fields::desc_ptr>(descPtrOffset);
    workItemDescriptor.write<Fields::wi_type>(workItemType);
    workItemDescriptor.write<Fields::wi_unit>(realTaskIndex.getTileIdx());
    workItemDescriptor.write<Fields::wi_sub_unit>(realTaskIndex.getListIdx());
    workItemDescriptor.write<Fields::next_workitem_idx>(nextWorkItemIdxValue);

    auto workItemDescriptorAttr = WorkItemAttr::get(rewriter.getContext(), std::move(workItemDescriptor));
    rewriter.create<NPUReg40XX::WorkItemOp>(origOp.getLoc(), origOp.getSymNameAttr(), origOp.getTaskTypeAttr(),
                                            origOp.getFirstTaskAttr(), workItemDescriptorAttr);
    rewriter.eraseOp(origOp);
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    return mlir::success();
}

//
// ConvertVPUASM2NPUReg40XXPass
//

class ConvertVPUASM2NPUReg40XXPass final : public impl::ConvertVPUASM2NPUReg40XXBase<ConvertVPUASM2NPUReg40XXPass> {
public:
    ConvertVPUASM2NPUReg40XXPass(Logger log, bool enableWLM) {
        Base::initLogger(log, Base::getArgumentName());
        _enableWLM = enableWLM;
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnModule() final;
    bool _enableWLM;
};

mlir::LogicalResult ConvertVPUASM2NPUReg40XXPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (wlmEnabled.hasValue()) {
        _enableWLM = wlmEnabled.getValue();
    }

    return mlir::success();
}

void ConvertVPUASM2NPUReg40XXPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto& ctx = getContext();
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp cnnOp;

    IE::CNNNetworkOp::getFromModule(moduleOp, cnnOp, netFunc);

    mlir::ConversionTarget target(ctx);

    target.addLegalDialect<NPUReg40XX::NPUReg40XXDialect>();
    target.addLegalDialect<VPUASM::VPUASMDialect>();

    auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    ELF::SymbolReferenceMap symRefMap(elfMain, true);

    mlir::RewritePatternSet patternNNDMA(&ctx);
    patternNNDMA.add<NNDMARewriter>(&ctx, _log, symRefMap);
    target.addIllegalOp<VPUASM::NNDMAOp>();
    if (mlir::failed(mlir::applyPartialConversion(netFunc, target, std::move(patternNNDMA)))) {
        signalPassFailure();
    }

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<BarrierRewriter>(&ctx, _log);
    patterns.add<M2IRewriter>(&ctx, _log, symRefMap);
    patterns.add<ActShaveRtRewriter>(&ctx, _log);
    patterns.add<ActKernelInvocationRewriter>(&ctx, _log, symRefMap);
    patterns.add<ActKernelRangeRewriter>(&ctx, _log, symRefMap);
    patterns.add<NNRTConfigRewriter>(&ctx, _log);
    patterns.add<ManagedBarrierRewriter>(&ctx, _log);
    patterns.add<MappedInferenceRewriter>(&ctx, _log);
    patterns.add<ManagedMappedInferenceRewriter>(&ctx, _log);
    patterns.add<MappedInferenceVersionRewriter>(&ctx, _log);
    patterns.add<WorkItemRewriter>(&ctx, _log);

    target.addIllegalOp<VPUASM::WorkItemOp>();
    target.addIllegalOp<VPUASM::MappedInferenceOp>();
    target.addIllegalOp<VPUASM::ConfigureBarrierOp>();
    target.addIllegalOp<VPUASM::M2IOp>();
    target.addIllegalOp<VPUASM::ActShaveRtOp>();
    target.addIllegalOp<VPUASM::ActKernelInvocationOp>();
    target.addIllegalOp<VPUASM::ActKernelRangeOp>();
    target.addIllegalOp<VPUASM::NNrtConfigOp>();
    target.addIllegalOp<VPUASM::ManagedBarrierOp>();
    target.addIllegalOp<VPUASM::ManagedMappedInferenceOp>();
    target.addIllegalOp<VPUASM::MappedInferenceVersionOp>();

    target.addDynamicallyLegalOp<VPUASM::PlatformInfoOp>([&](VPUASM::PlatformInfoOp op) {
        return VPU::getArch(op.getOperation()) != VPU::ArchKind::UNKNOWN;
    });

    if (mlir::failed(mlir::applyPartialConversion(netFunc, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertVPUASM2NPUReg40XXPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUASM2NPUReg40XXPass(Logger log, bool enableWLM) {
    return std::make_unique<ConvertVPUASM2NPUReg40XXPass>(log, enableWLM);
}
