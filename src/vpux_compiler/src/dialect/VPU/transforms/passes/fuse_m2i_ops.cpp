//
// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/config/IR/resources.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/m2i_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/enums.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_FUSEM2IOPS
#define GEN_PASS_DEF_FUSEM2IOPS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

mlir::FailureOr<M2iColorFmt> getM2iPlanarOutFmt(mlir::Type oType, IE::ColorFmt cscOutFmt) {
    if (oType.isF16()) {
        if (cscOutFmt == IE::ColorFmt::RGB || cscOutFmt == IE::ColorFmt::BGR) {
            return M2iColorFmt::PL_FP16_RGB;
        }
    } else if (oType.isUnsignedInteger(8)) {
        if (cscOutFmt == IE::ColorFmt::RGB || cscOutFmt == IE::ColorFmt::BGR) {
            return M2iColorFmt::PL_RGB24;
        }
    }
    return mlir::failure();
}

// Check Permute transforms Interleaved->Planar: NHWC(d0,d1,d2,d3) -> NCHW(d0,d3,d1,d2)
inline bool checkPerm(mlir::AffineMap memPerm, mlir::MLIRContext* ctx) {
    const SmallVector<uint32_t> order{0, 3, 1, 2};
    const auto map = mlir::AffineMap::getPermutationMap(ArrayRef(order), ctx);
    return memPerm == map;
}

// ======================================================================================
// FuseM2iCscResizePl (Planar output)
//   NV12/I420(u8) -> [CSC -> [ConvertU8toF16] -> Resize -> Permute] -> Planar_FP16_RGB/BGR
//   NV12/I420(u8) -> [CSC -> Resize -> [ConvertU8toF16] -> Permute] -> Planar_FP16_RGB/BGR
//   NV12/I420(u8) -> [CSC -------------------> Resize -> Permute] -> Planar_U8_RGB/BGR
//   IL_RGB888     -> [ Convert -> Resize -> Permute] -> Planar_FP16_RGB
//   IL_RGB888     -> [ Resize -> Convert -> Permute] -> Planar_FP16_RGB
//   IL_RGB888     -> [            Resize -> Permute] -> Planar_U8_RGB/BGR
//

class FuseM2iCscResizePl final : public mlir::OpRewritePattern<VPU::MemPermuteOp> {
public:
    FuseM2iCscResizePl(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::MemPermuteOp>(ctx, benefitHigh), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseM2iCscResizePl::matchAndRewrite(VPU::MemPermuteOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (!checkPerm(origOp.getMemPerm(), getContext())) {
        return mlir::failure();
    }

    auto postInterpInVal = origOp.getInput();
    auto postConvertOp = origOp.getInput().getDefiningOp<VPU::ConvertOp>();
    if (postConvertOp) {
        if (!mlir::cast<vpux::NDTypeInterface>(postConvertOp.getInput().getType())
                     .getElementType()
                     .isUnsignedInteger(8) ||
            !mlir::cast<vpux::NDTypeInterface>(postConvertOp.getOutput().getType()).getElementType().isF16()) {
            _log.trace("[{0}] Only convert from u8 to fp16 accepted", getDebugName());
            return mlir::failure();
        }
        postInterpInVal = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(postConvertOp.getInput());
    }

    mlir::Value sclInput;
    mlir::ArrayAttr sclSizes;
    mlir::ArrayAttr sclAxes;
    VPU::M2iInterp interpMode;
    if (auto m2iScl = postInterpInVal.getDefiningOp<VPU::M2IResizeOp>()) {
        sclInput = m2iScl.getInput();
        sclSizes = m2iScl.getSizes();
        sclAxes = m2iScl.getAxes();
        interpMode = m2iScl.getInterp();
    } else if (auto interp = postInterpInVal.getDefiningOp<VPU::InterpolateOp>()) {
        const auto logCb = [&](const llvm::formatv_object_base& msg) {
            std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
        };
        if (!VPU::isM2IResizeSupported<VPU::InterpolateOp>(interp, logCb, false /*checkFp16Interleaved*/)) {
            return mlir::failure();
        }
        sclInput = interp.getInput();
        sclSizes = interp.getSizesAttr().value();
        sclAxes = interp.getAxesAttr().value();
        interpMode = VPU::IEtoM2iInterpMode(interp.getAttr().getMode().getValue());
    } else {
        _log.trace("[{0}] FuseM2iCscResizePl pattern not matching", getDebugName());
        return mlir::failure();
    }

    // optional VPU::ConvertOp
    VPU::M2IColorConvertOp m2iCsc;
    auto preConvertOp = sclInput.getDefiningOp<VPU::ConvertOp>();
    if (preConvertOp) {
        if (!mlir::cast<vpux::NDTypeInterface>(preConvertOp.getInput().getType())
                     .getElementType()
                     .isUnsignedInteger(8) ||
            !mlir::cast<vpux::NDTypeInterface>(preConvertOp.getOutput().getType()).getElementType().isF16()) {
            _log.trace("[{0}] Only convert from u8 to fp16 accepted", getDebugName());
            return mlir::failure();
        }
        m2iCsc = preConvertOp.getInput().getDefiningOp<VPU::M2IColorConvertOp>();
    } else {
        m2iCsc = sclInput.getDefiningOp<VPU::M2IColorConvertOp>();
    }

    bool inColorOrder{}, outColorOrder{}, inLumaOrder{}, outLumaOrder{};
    VPU::M2iColorFmt iFmt, oFmt;
    mlir::Value patternInputValue{};
    if (m2iCsc == nullptr) {
        _log.trace("[{0}] Pattern: [Convert] -> Resize -> Permute", getDebugName());
        const auto axes = parseIntArrayAttr<int64_t>(sclAxes);
        const auto axesSize = axes.size();
        if (!axesSize) {
            _log.trace("[{0}] Axes attribute has no values", getDebugName());
            return mlir::failure();
        }
        const auto lastAxis = axes[axesSize - 1];
        const auto lastDim = mlir::cast<vpux::NDTypeInterface>(sclInput.getType()).getRank() - 1;
        const bool isScalingInterleaved = (lastAxis != lastDim);
        if (!isScalingInterleaved) {
            _log.trace("[{0}] Pattern assumes an interleaved scaling", getDebugName());
            return mlir::failure();
        }
        iFmt = VPU::M2iColorFmt::IL_RGB888;
        if (preConvertOp == nullptr && postConvertOp == nullptr) {
            // no conversion op
            oFmt = M2iColorFmt::PL_RGB24;
            patternInputValue = sclInput;
        } else {
            oFmt = VPU::M2iColorFmt::PL_FP16_RGB;
            patternInputValue = preConvertOp.getInput();
        }
    } else {
        inColorOrder = getM2iColorOrderReg(m2iCsc.getInFmtAttr().getValue());
        outColorOrder = getM2iColorOrderReg(m2iCsc.getOutFmtAttr().getValue());
        inLumaOrder = getM2iLumaOrderReg(m2iCsc.getInFmtAttr().getValue());
        outLumaOrder = getM2iLumaOrderReg(m2iCsc.getOutFmtAttr().getValue());
        iFmt = IEtoM2iColorFmt(m2iCsc.getInFmtAttr().getValue());
        const auto oType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getElementType();
        const auto cscOutFmt = m2iCsc.getOutFmtAttr().getValue();

        const auto res = getM2iPlanarOutFmt(oType, cscOutFmt);
        if (mlir::failed(res)) {
            _log.trace("[{0}] Output format not valid", getDebugName());
            return mlir::failure();
        }
        oFmt = res.value();
        patternInputValue = m2iCsc.getInput();
    }

    auto m2iOp = rewriter.create<VPU::M2ITaskOp>(origOp->getLoc(), origOp.getType(), patternInputValue,
                                                 (m2iCsc != nullptr), false, iFmt, oFmt, inColorOrder, outColorOrder,
                                                 inLumaOrder, outLumaOrder, sclSizes, sclAxes, nullptr, interpMode);
    rewriter.replaceOp(origOp, m2iOp.getOutput());

    return mlir::success();
}  // namespace

// ======================================================================================
// FuseM2iTask2Pl (Planar output)
//     [M2ITaskOp(u8_IL) -> Permute] becomes [M2ITaskOp(u8_PL)]
//     [M2ITaskOp(u8_IL) -> Convert -> Permute] becomes [M2ITaskOp(fp16_PL)]

class FuseM2iTask2Pl final : public mlir::OpRewritePattern<VPU::MemPermuteOp> {
public:
    FuseM2iTask2Pl(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::MemPermuteOp>(ctx, benefitHigh), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseM2iTask2Pl::matchAndRewrite(VPU::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (!checkPerm(origOp.getMemPerm(), getContext())) {
        return mlir::failure();
    }

    auto postTaskInVal = origOp.getInput();
    // Optional ConvertOp
    auto opConvert = origOp.getInput().getDefiningOp<VPU::ConvertOp>();
    if (opConvert) {
        if (!mlir::cast<vpux::NDTypeInterface>(opConvert.getInput().getType()).getElementType().isUnsignedInteger(8) ||
            !mlir::cast<vpux::NDTypeInterface>(opConvert.getOutput().getType()).getElementType().isF16()) {
            _log.trace("[{0}] Only convert from u8 to fp16 accepted in pattern", getDebugName());
            return mlir::failure();
        }
        postTaskInVal = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(opConvert.getInput());
    }

    auto task = postTaskInVal.getDefiningOp<VPU::M2ITaskOp>();
    if (task == nullptr) {
        return mlir::failure();
    }

    const auto iFmt = task.getInFmtAttr().getValue();
    const auto taskOutFmt = task.getOutFmtAttr().getValue();

    M2iColorFmt oFmt;
    if (taskOutFmt == M2iColorFmt::IL_RGB888) {
        if (opConvert) {
            oFmt = M2iColorFmt::PL_FP16_RGB;
        } else {
            oFmt = M2iColorFmt::PL_RGB24;
        }
    } else {
        return mlir::failure();
    }

    auto m2iOp = rewriter.create<VPU::M2ITaskOp>(origOp->getLoc(), origOp.getType(), task.getInput(), task.getDoCsc(),
                                                 task.getDoNorm(), iFmt, oFmt, task.getChromaInReverseChannels(),
                                                 task.getChromaOutReverseChannels(), task.getLumaInReverseChannels(),
                                                 task.getLumaOutReverseChannels(), task.getSizesAttr(),
                                                 task.getAxesAttr(), task.getNormAttr(), task.getInterp());
    rewriter.replaceOp(origOp, m2iOp.getOutput());

    return mlir::success();
}

// ======================================================================================
// FuseM2iCscResizeIl (Interleaved output)
//   NV12/I420(u8) -> [CSC -> Resize] -> Interleaved_U8_RGB/BGR
//

class FuseM2iCscResizeIl final : public mlir::OpRewritePattern<VPU::M2IResizeOp> {
public:
    FuseM2iCscResizeIl(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::M2IResizeOp>(ctx, benefitLow), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::M2IResizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseM2iCscResizeIl::matchAndRewrite(VPU::M2IResizeOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto m2iScl = origOp;
    auto m2iCsc = origOp.getInput().getDefiningOp<VPU::M2IColorConvertOp>();
    if (m2iCsc == nullptr) {
        return mlir::failure();
    }

    auto inColorOrder = getM2iColorOrderReg(m2iCsc.getInFmtAttr().getValue());
    auto outColorOrder = getM2iColorOrderReg(m2iCsc.getOutFmtAttr().getValue());
    auto inLumaOrder = getM2iLumaOrderReg(m2iCsc.getInFmtAttr().getValue());
    auto outLumaOrder = getM2iLumaOrderReg(m2iCsc.getOutFmtAttr().getValue());

    const auto iFmt = IEtoM2iColorFmt(m2iCsc.getInFmtAttr().getValue());
    const auto oFmt = IEtoM2iColorFmt(m2iCsc.getOutFmtAttr().getValue());

    auto m2iOp = rewriter.create<VPU::M2ITaskOp>(origOp->getLoc(), origOp.getType(), m2iCsc.getInput(), true, false,
                                                 iFmt, oFmt, inColorOrder, outColorOrder, inLumaOrder, outLumaOrder,
                                                 m2iScl.getSizes(), m2iScl.getAxes(), nullptr, m2iScl.getInterp());
    rewriter.replaceOp(origOp, m2iOp.getOutput());

    return mlir::success();
}

// ======================================================================================
// FuseM2iCscPl (Planar output)
//   NV12/I420(u8) -> [CSC -> {convertU8F16} -> Permute] -> Planar_U8/F16_RGB/BGR
//

class FuseM2iCscPl final : public mlir::OpRewritePattern<VPU::MemPermuteOp> {
public:
    FuseM2iCscPl(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::MemPermuteOp>(ctx, benefitLow), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseM2iCscPl::matchAndRewrite(VPU::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (!checkPerm(origOp.getMemPerm(), getContext())) {
        return mlir::failure();
    }

    auto opConvert = origOp.getInput().getDefiningOp<VPU::ConvertOp>();  // optional

    VPU::M2IColorConvertOp m2iCsc;
    if (opConvert == nullptr) {
        m2iCsc = origOp.getInput().getDefiningOp<VPU::M2IColorConvertOp>();
    } else {
        if (!mlir::cast<vpux::NDTypeInterface>(opConvert.getOutput().getType()).getElementType().isF16()) {
            _log.trace("[{0}] Only convert from u8 to fp16 accepted", getDebugName());
            return mlir::failure();
        }
        m2iCsc = opConvert.getInput().getDefiningOp<VPU::M2IColorConvertOp>();
    }

    if (m2iCsc == nullptr) {
        _log.trace("[{0}] FuseM2iCscPl Pattern not matching", getDebugName());
        return mlir::failure();
    }

    auto inColorOrder = getM2iColorOrderReg(m2iCsc.getInFmtAttr().getValue());
    auto outColorOrder = getM2iColorOrderReg(m2iCsc.getOutFmtAttr().getValue());
    auto inLumaOrder = getM2iLumaOrderReg(m2iCsc.getInFmtAttr().getValue());
    auto outLumaOrder = getM2iLumaOrderReg(m2iCsc.getOutFmtAttr().getValue());

    const auto iFmt = IEtoM2iColorFmt(m2iCsc.getInFmtAttr().getValue());
    const auto oType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getElementType();
    const auto cscOutFmt = m2iCsc.getOutFmtAttr().getValue();

    M2iColorFmt oFmt;
    auto res = getM2iPlanarOutFmt(oType, cscOutFmt);
    if (failed(res)) {
        return mlir::failure();
    } else {
        oFmt = res.value();
    }

    auto m2iOp = rewriter.create<VPU::M2ITaskOp>(origOp->getLoc(), origOp.getType(), m2iCsc.getInput(), true, false,
                                                 iFmt, oFmt, inColorOrder, outColorOrder, inLumaOrder, outLumaOrder,
                                                 nullptr, nullptr, nullptr);
    rewriter.replaceOp(origOp, m2iOp.getOutput());

    return mlir::success();
}

// ======================================================================================

using InterpConfigs = std::tuple<mlir::ArrayAttr, mlir::ArrayAttr, VPU::M2iInterp>;
mlir::FailureOr<InterpConfigs> getResizeConfigs(VPU::M2IResizeOp resizeOp, LogCb) {
    return InterpConfigs{resizeOp.getSizes(), resizeOp.getAxes(), resizeOp.getInterp()};
}

mlir::FailureOr<InterpConfigs> getResizeConfigs(VPU::InterpolateOp interp, LogCb logCb) {
    if (!VPU::isM2IResizeSupported<VPU::InterpolateOp>(interp, logCb, true /*checkFp16Interleaved*/)) {
        return mlir::failure();
    }

    auto interpMode = VPU::IEtoM2iInterpMode(interp.getAttr().getMode().getValue());
    return InterpConfigs{interp.getSizesAttr().value(), interp.getAxesAttr().value(), interpMode};
}

// FuseNormBase
template <typename ConcreteOp>
class FuseNormBase : public mlir::OpRewritePattern<VPU::M2INormOp> {
public:
    FuseNormBase(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::M2INormOp>(ctx, benefitLow), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPU::M2INormOp normOp, mlir::PatternRewriter& rewriter) const final;

protected:
    virtual mlir::FailureOr<ConcreteOp> getResizeOp(VPU::M2INormOp origOp) const = 0;
    virtual mlir::Operation* getParent(VPU::M2INormOp normOp) const = 0;
    virtual mlir::Operation* getChild(VPU::M2INormOp normOp) const = 0;

protected:
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult FuseNormBase<ConcreteOp>::matchAndRewrite(VPU::M2INormOp normOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), normOp->getName(), normOp->getLoc());

    if (!mlir::cast<vpux::NDTypeInterface>(normOp.getInput().getType()).getElementType().isF16()) {
        return mlir::failure();
    }

    auto maybeResizeOp = getResizeOp(normOp);
    if (mlir::failed(maybeResizeOp)) {
        _log.trace("[{0}] Pattern not matching", this->getDebugName());
        return mlir::failure();
    }

    auto resizeOp = maybeResizeOp.value();
    const auto logCb = [&](const llvm::formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, normOp, "[{0}] {1}", this->getDebugName(), msg.str());
    };

    // here resizeOp is either a (nonnull) VPU::M2IResizeOp or a VPU::InterpolateOp
    auto maybeConfig = getResizeConfigs(resizeOp, logCb);
    if (mlir::failed(maybeConfig)) {
        _log.trace("[{0}] Error in retrieving resize configurations", this->getDebugName());
        return mlir::failure();
    }

    mlir::ArrayAttr resizeSizes, resizeAxes;
    VPU::M2iInterp interpMode;
    std::tie(resizeSizes, resizeAxes, interpMode) = maybeConfig.value();
    bool inColorOrder{false}, outColorOrder{false}, inLumaOrder{false}, outLumaOrder{false};
    // no openVino operator works on yuv, which means here the color space has to be already rgb
    const auto iFmt = VPU::M2iColorFmt::PL_FP16_RGB;
    const auto oFmt = VPU::M2iColorFmt::PL_FP16_RGB;
    auto coeffsAttr = VPU::getM2INormCoeffsAttr(normOp->getContext(), normOp);

    auto parentOp = getParent(normOp);
    auto childOp = getChild(normOp);
    auto m2iOp = rewriter.create<VPU::M2ITaskOp>(
            childOp->getLoc(), childOp->getResult(0).getType(), parentOp->getOperand(0), false, true, iFmt, oFmt,
            inColorOrder, outColorOrder, inLumaOrder, outLumaOrder, resizeSizes, resizeAxes, coeffsAttr, interpMode);

    rewriter.replaceOp(childOp, m2iOp.getOutput());

    return mlir::success();
}

// ======================================================================================
template <typename ConcreteOp>
class FuseParentAndNorm final : public FuseNormBase<ConcreteOp> {
public:
    FuseParentAndNorm(mlir::MLIRContext* ctx, Logger log): FuseNormBase<ConcreteOp>(ctx, log) {
    }

protected:
    using FuseNormBase<ConcreteOp>::_log;

protected:
    mlir::FailureOr<ConcreteOp> getResizeOp(VPU::M2INormOp normOp) const override {
        auto parentOp = normOp.getInput().getDefiningOp();
        if (parentOp == nullptr || !parentOp->getResult(0).hasOneUse()) {
            _log.trace("[{0}] Parent op must have one use", this->getDebugName());
            return mlir::failure();
        }

        if (auto resize = mlir::dyn_cast<ConcreteOp>(parentOp)) {
            return resize;
        }

        return mlir::failure();
    }

    mlir::Operation* getParent(VPU::M2INormOp normOp) const override {
        return normOp.getInput().getDefiningOp();
    }

    mlir::Operation* getChild(VPU::M2INormOp normOp) const override {
        return normOp;
    }
};

template <typename ConcreteOp>
class FuseNormAndChild final : public FuseNormBase<ConcreteOp> {
public:
    FuseNormAndChild(mlir::MLIRContext* ctx, Logger log): FuseNormBase<ConcreteOp>(ctx, log) {
    }

protected:
    using FuseNormBase<ConcreteOp>::_log;

protected:
    mlir::FailureOr<ConcreteOp> getResizeOp(VPU::M2INormOp origOp) const override {
        auto result = origOp.getResult();
        if (!result.hasOneUse()) {
            _log.trace("[{0}] Norm op must have one use", this->getDebugName());
            return mlir::failure();
        }

        if (auto resize = mlir::dyn_cast<ConcreteOp>(*result.getUsers().begin())) {
            return resize;
        }

        return mlir::failure();
    }

    mlir::Operation* getParent(VPU::M2INormOp normOp) const override {
        return normOp;
    }

    mlir::Operation* getChild(VPU::M2INormOp normOp) const override {
        return *normOp.getResult().getUsers().begin();
    }
};

// ======================================================================================
// FuseM2iTaskNormBase

template <typename ConcreteOp>
class FuseM2ITaskOpNormBase : public mlir::OpRewritePattern<ConcreteOp> {
public:
    FuseM2ITaskOpNormBase(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx, benefitLow), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

protected:
    virtual bool isPatternValid(ConcreteOp origOp) const = 0;
    virtual VPU::M2ITaskOp getM2ITaskOp(ConcreteOp origOp) const = 0;
    virtual VPU::M2INormOp getNormOp(ConcreteOp origOp) const = 0;

    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult FuseM2ITaskOpNormBase<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    if (!mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getElementType().isF16()) {
        return mlir::failure();
    }

    bool patternValid = isPatternValid(origOp);
    auto m2iTaskOp = getM2ITaskOp(origOp);
    auto normOp = getNormOp(origOp);
    if (!patternValid || m2iTaskOp == nullptr || normOp == nullptr) {
        _log.trace("[{0}] Pattern not matching", this->getDebugName());
        return mlir::failure();
    }

    auto parentOp = origOp.getInput().getDefiningOp();
    if (!parentOp->getResult(0).hasOneUse()) {
        _log.trace("[{0}] Parent op must have one use", this->getDebugName());
        return mlir::failure();
    }

    auto* cloneOp = rewriter.clone(*m2iTaskOp);
    auto newM2iTaskOp = llvm::dyn_cast_or_null<VPU::M2ITaskOp>(cloneOp);
    if (newM2iTaskOp == nullptr) {
        _log.trace("[{0}] Problem cloning m2iTaskOp", this->getDebugName());
        return mlir::failure();
    }

    newM2iTaskOp->setOperand(0, parentOp->getOperand(0));
    newM2iTaskOp->getResult(0).setType(origOp.getOutput().getType());

    if (!newM2iTaskOp.getDoNorm()) {
        // the existing M2ITask doesn't have normalisation operations
        newM2iTaskOp.setDoNorm(true);
        newM2iTaskOp.setNormAttr(VPU::getM2INormCoeffsAttr(origOp->getContext(), normOp));
    } else {
        // existing M2ITask already has some normalisation coefficients: it should be possible to compose the two
        // y1 = A1 * ((x1 - B1) / C1 ) + D1; y2 = A2 * ((y1 - B2) / C2 ) + D2 = A2 * (((A1*((x -
        // B1)/C1)+D1)-B2)/C2)+D2 = A1*A2*(x-B1)/(C1*C2) + D2 + (D1-B2)/C2

        // composed operation would have A = A1*A2, B = B1, C = C1*C2, D = D2 + (D1-B2)/C2
        // Only concern is the effect on precision: for the time being don't enable this operation composition
        _log.trace("[{0}] M2ITask already has a normalisation operation enabled: don't fuse with M2INormOp",
                   this->getDebugName());
        return mlir::failure();
    }

    rewriter.replaceOp(origOp, newM2iTaskOp.getOutput());

    return mlir::success();
}

// ======================================================================================
// FuseM2iTaskNorm
//   [M2ITask -> Norm] -> [M2ITask]
class FuseM2ITaskNorm final : public FuseM2ITaskOpNormBase<VPU::M2INormOp> {
public:
    FuseM2ITaskNorm(mlir::MLIRContext* ctx, Logger log): FuseM2ITaskOpNormBase<VPU::M2INormOp>(ctx, log) {
    }

    virtual bool isPatternValid(VPU::M2INormOp origOp) const override {
        auto parentOp = origOp.getInput().getDefiningOp();
        return mlir::isa_and_nonnull<VPU::M2ITaskOp>(parentOp);
    }
    virtual VPU::M2ITaskOp getM2ITaskOp(VPU::M2INormOp origOp) const override {
        auto parentOp = origOp.getInput().getDefiningOp();
        return llvm::dyn_cast_or_null<VPU::M2ITaskOp>(parentOp);
    }
    virtual VPU::M2INormOp getNormOp(VPU::M2INormOp origOp) const override {
        return origOp;
    }
};

// ======================================================================================
// FuseM2INormTask
//   [Norm -> M2ITask] -> [M2ITask]
class FuseM2INormTask final : public FuseM2ITaskOpNormBase<VPU::M2ITaskOp> {
public:
    FuseM2INormTask(mlir::MLIRContext* ctx, Logger log): FuseM2ITaskOpNormBase<VPU::M2ITaskOp>(ctx, log) {
    }

    virtual bool isPatternValid(VPU::M2ITaskOp origOp) const override {
        auto parentOp = origOp.getInput().getDefiningOp();
        return mlir::isa_and_nonnull<VPU::M2INormOp>(parentOp);
    }
    virtual VPU::M2ITaskOp getM2ITaskOp(VPU::M2ITaskOp origOp) const override {
        return origOp;
    }
    virtual VPU::M2INormOp getNormOp(VPU::M2ITaskOp origOp) const override {
        auto parentOp = origOp.getInput().getDefiningOp();
        return llvm::dyn_cast_or_null<VPU::M2INormOp>(parentOp);
    }
};

//
// FuseM2IOpsPass
//

class FuseM2IOpsPass final : public VPU::impl::FuseM2IOpsBase<FuseM2IOpsPass> {
public:
    explicit FuseM2IOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FuseM2IOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseM2iCscResizePl>(&ctx, _log);
    patterns.add<FuseM2iCscResizeIl>(&ctx, _log);
    patterns.add<FuseM2iCscPl>(&ctx, _log);
    patterns.add<FuseM2iTask2Pl>(&ctx, _log);
    patterns.add<FuseParentAndNorm<VPU::M2IResizeOp>>(&ctx, _log);
    patterns.add<FuseParentAndNorm<VPU::InterpolateOp>>(&ctx, _log);
    patterns.add<FuseNormAndChild<VPU::M2IResizeOp>>(&ctx, _log);
    patterns.add<FuseNormAndChild<VPU::InterpolateOp>>(&ctx, _log);
    patterns.add<FuseM2ITaskNorm>(&ctx, _log);
    patterns.add<FuseM2INormTask>(&ctx, _log);

    mlir::ConversionTarget target(ctx);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createFuseM2IOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createFuseM2IOpsPass(Logger log) {
    return std::make_unique<FuseM2IOpsPass>(log);
}
